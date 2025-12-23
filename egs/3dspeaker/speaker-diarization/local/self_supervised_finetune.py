# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
Self-supervised learning pipeline for speaker diarization.
This script implements iterative fine-tuning of speaker embedding models using
pseudo-labels.

Pipeline:
- Round 0 (part 0): Initial clustering with HMM correction
- Round 0 (part 1): Fine-tune embedding model with pseudo-labels
- Round 0 (Part 1): Extract embeddings, cluster, evaluate
- Round 1+: Iterate until convergence or max rounds
"""

import os, sys, time, shutil, subprocess, copy
import json, argparse, random, pickle, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler, SequentialSampler

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from speakerlab.utils.builder import build
from speakerlab.utils.fileio import load_audio
from speakerlab.utils.config import yaml_config_loader, Config
from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
from extract_diar_embeddings import update_conf
from vision_tools.face_recognition_pytorch import IR_101


parser = argparse.ArgumentParser(description='Self-supervised fine-tuning for speaker diarization')
# Clustering parameters
parser.add_argument('--conf', required=True, type=str, help='Config file')
parser.add_argument('--wavs', required=True, type=str, help='Wav list file')
parser.add_argument('--cluster_type', default='audio_only', type=str, help='Clustering type, support "audio_only" and "audio_vision"')
parser.add_argument('--audio_embs_dir', required=True, type=str, help='Initial audio embeddings directory')
parser.add_argument('--visual_embs_dir', required=True, type=str, help='Visual embeddings directory')
parser.add_argument('--result_dir', required=True, type=str, help='Result directory')
# HMM parameters
parser.add_argument('--use_hmm_smoothing', action='store_true', help='Use HMM smoothing in iterations')
parser.add_argument('--fix_mf', action='store_true', help='Fix key frame visual cluster labels during HMM smoothing')
parser.add_argument('--hmm_visual_info_type', default='vad+mid_frame', type=str, help='Visual information type, support "", "vad", "mid_frame", "vad+mid_frame"')
parser.add_argument('--unreliable_pp', default=100.0, type=float, help='Percentage of unreliable segments to be smoothed, default 100.0 (all segments)')
# Speaker annotation file
parser.add_argument('--speaker_anno_file', required=True, type=str, help='Speaker annotation xlsx file')
parser.add_argument('--face_anno_file', required=True, type=str, help='Face annotation xlsx file')
parser.add_argument('--face_pretrained_model', required=True, type=str, help='Path to face pretrained model')
parser.add_argument('--midframe_face_dir', required=True, type=str, help='Dir of midframe_faces')

parser.add_argument('--speaker_model_id', default=None, help='Speaker model id in modelscope')
parser.add_argument('--speaker_pretrained_model', default=None, type=str, help='Path of local pretrained model')    # Don't need to set, system will detect from cache(and download if necessary)
parser.add_argument('--subseg_json', required=True, type=str, help='Sub-segments json file')

# Self-supervised learning parameters
parser.add_argument('--finetune_lr', default=0.001, type=float, help='Fine-tuning learning rate')
parser.add_argument('--finetune_batch_size', default=64, type=int, help='Fine-tuning batch size')
parser.add_argument('--unfrozen_layers_num', default=2, type=int, help='Number of layers to unfreeze (from top) during fine-tuning')
parser.add_argument('--warmup_epochs_num', default=2, type=int, help='Warmup epochs for classifier head')
parser.add_argument('--max_rounds', default=10, type=int, help='Maximum self-supervised learning rounds')
parser.add_argument('--max_finetune_epochs', default=5, type=int, help='Number of max epochs for each fine-tuning stage')
parser.add_argument('--early_stop_patience_round', default=5, type=int, help='Early stopping patience for rounds')
parser.add_argument('--early_stop_patience_epoch', default=5, type=int, help='Early stopping patience for epochs')
parser.add_argument('--from_preds', action='store_true', help='Use local predictions from classifier model instead of clustering to generate pseudo labels')
parser.add_argument('--use_hidfeat', action='store_true', help='Use hidden features from embedding model to construct dataset instead of original features')

# Distributed training
parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')
parser.add_argument('--seed', default=1234, type=int, help='Random seed')


class PseudoLabelDataset(Dataset):
    """
    Dataset for fine-tuning with pseudo-labels from pseudo labels.
    """
    def __init__(self, subseg_json, pseudo_labels_json, feature_extractor, use_hidfeat_flag = False, subseg_hidfeat_dic=None):
        """
        Args:
            subseg_json: Path to sub-segments json file
            pseudo_labels_json: Path to pseudo labels json file
            feature_extractor: Feature extraction object
            use_hidfeat_flag: If True, use hidden features from embedding model
            subseg_hidfeat_dic: Dictionary of hidden features for sub-segments (if use_hidfeat_flag is True)
        """
        self.feature_extractor = feature_extractor
        self.use_hidfeat_flag = use_hidfeat_flag
        self.subseg_hidfeat_dic = subseg_hidfeat_dic
        
        # Load sub-segments info
        with open(subseg_json, 'r') as f:
            self.subseg_info = json.load(f)
        
        # 不再过滤-1标签，将所有样本都用于训练
        self.sample_ids = list(self.subseg_info.keys())
        # 设置各个sample的标签
        self.pseudo_labels = {}
        self.unique_labels = []
        update_flag = self.reset_labels(pseudo_labels_json)
        
        if not self.use_hidfeat_flag:
            # Get all wav data to speed up loading
            obj_fs = self.feature_extractor.sample_rate
            self.wav_paths = list(set([self.subseg_info[sid]['file'] for sid in self.sample_ids]))
            self.wav_dat_dic = {wav_path: load_audio(wav_path, obj_fs=obj_fs) for wav_path in self.wav_paths}
            self.subseg_wav_dic = {sid: self.wav_dat_dic[self.subseg_info[sid]['file']][0, int(self.subseg_info[sid]['start']*obj_fs):int(self.subseg_info[sid]['stop']*obj_fs)].unsqueeze(0) for sid in self.sample_ids}    # each elements is (1, num_samples_i))
            del self.wav_paths, self.wav_dat_dic

            # 预先提取特征并记录每个sample的特征帧数
            self.subseg_feat_dic = {}
            self.sample_lengths = {}
            for sid in self.sample_ids:
                waveform = self.subseg_wav_dic[sid]
                feat = torch.vmap(self.feature_extractor)(waveform.unsqueeze(0)).squeeze(0) # mel feature of shape [num_frames, n_mels]
                self.subseg_feat_dic[sid] = feat
                self.sample_lengths[sid] = feat.shape[0]  # 记录特征帧数而非音频采样点数
        else:
            assert self.subseg_hidfeat_dic is not None, "Hidden features dictionary must be provided when use_hidfeat_flag is True!"

    def reset_labels(self, pseudo_labels_json):
        """
        Update pseudo-labels from a new pseudo labels json file.
        Args:
            pseudo_labels_json: Path to new pseudo labels json file
        """
        # Load pseudo-labels
        with open(pseudo_labels_json, 'r') as f:
            pseudo_labels = json.load(f)
        
        # Assert that the keys in subseg_info and pseudo_labels match
        assert set(self.subseg_info.keys()) == set(pseudo_labels.keys()), \
            "Keys in subseg_info and pseudo_labels do not match!"
        
        # Create label encoder (map cluster IDs to continuous class indices)
        unique_labels = sorted(list(set([pseudo_labels[sid] for sid in self.sample_ids])))
        if set(unique_labels).issubset(set(self.unique_labels)):
            # don't update unique_labels, label2idx, idx2label, num_classes to reuse classifier
            # some class may be missing in new pseudo labels
            update_flag = False
        else:
            update_flag = True
            # update unique_labels
            self.unique_labels = unique_labels
            self.label2idx = {label: idx for idx, label in enumerate(self.unique_labels)}
            self.idx2label = {idx: label for label, idx in self.label2idx.items()}
            self.num_classes = len(self.unique_labels)
        
        self.pseudo_labels = pseudo_labels
        self.class_weights = self.get_class_weights()
        self.label_weights = {self.idx2label[idx]: self.class_weights[idx].item() for idx in range(self.num_classes)}
        
        print(f"[INFO] Created dataset with {len(self.sample_ids)} samples and {self.num_classes} classes")
        print(f"[INFO] Label weights: {self.label_weights}")
        return update_flag
    
    def get_class_weights(self):
        """
        Compute class weights inversely proportional to the square root of class frequencies
        Returns:
            class_weights: Tensor of shape [num_classes]
        """
        class_counts = np.zeros(self.num_classes, dtype=np.int64)
        for sid in self.sample_ids:
            cluster_label = self.pseudo_labels[sid]
            class_idx = self.label2idx[cluster_label]
            class_counts[class_idx] += 1
        class_valid_flag = class_counts > 0
        valid_count = np.sum(class_valid_flag)
        
        class_weights = np.zeros(self.num_classes, dtype=np.float32)
        class_weights[class_valid_flag] = 1.0 / (class_counts[class_valid_flag])**0.5
        class_weights = class_weights / np.sum(class_weights) * valid_count  # 归一化
        return torch.tensor(class_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        subseg_id = self.sample_ids[index]
        # Get pseudo-label
        cluster_label = self.pseudo_labels[subseg_id]
        class_idx = self.label2idx[cluster_label]

        if not self.use_hidfeat_flag: # use raw pre-extracted features
            feat = self.subseg_feat_dic[subseg_id]
            length = self.sample_lengths[subseg_id]
            return feat, class_idx, length
        else:  # Use hidden features
            feat = self.subseg_hidfeat_dic[subseg_id]
            return feat, class_idx

class PseudoLabelFaceDataset(Dataset):
    """
    Dataset for fine-tuning face recognition model with pseudo-labels.
    """
    def __init__(self, pseudo_labels_json, dataset_dir):
        """
        Args:
            pseudo_labels_json: Path to pseudo labels json file (format: {"E01-1_0": 0, ...})
            dataset_dir: Directory containing mid-frame face images organized by episode
        """
        self.dataset_dir = dataset_dir
        self.pseudo_labels = {}
        self.sample_ids = []
        self.unique_labels = []
        update_flag = self.reset_labels(pseudo_labels_json)

    def reset_labels(self, pseudo_labels_json):
        """
        Update pseudo-labels from a new pseudo labels json file.
        Args:
            pseudo_labels_json: Path to new pseudo labels json file
        """
        # Load pseudo-labels
        with open(pseudo_labels_json, 'r') as f:
            pseudo_labels = json.load(f)
        self.sample_ids = list(pseudo_labels.keys())
        
        # Create label encoder (map cluster IDs to continuous class indices)
        unique_labels = sorted(list(set([pseudo_labels[sid] for sid in self.sample_ids])))
        if set(unique_labels).issubset(set(self.unique_labels)):
            # don't update unique_labels, label2idx, idx2label, num_classes to reuse classifier
            # some class may be missing in new pseudo labels
            update_flag = False
        else:
            update_flag = True
            self.unique_labels = unique_labels
            self.label2idx = {label: idx for idx, label in enumerate(self.unique_labels)}
            self.idx2label = {idx: label for label, idx in self.label2idx.items()}
            self.num_classes = len(self.unique_labels)
        
        self.pseudo_labels = pseudo_labels
        self.class_weights = self.get_class_weights()
        self.label_weights = {self.idx2label[idx]: self.class_weights[idx].item() for idx in range(self.num_classes)}
        
        print(f"[INFO] Created face dataset with {len(self.sample_ids)} samples and {self.num_classes} classes")
        print(f"[INFO] Face label weights: {self.label_weights}")
        return update_flag
    
    def get_class_weights(self):
        """
        Compute class weights inversely proportional to the square root of class frequencies
        Returns:
            class_weights: Tensor of shape [num_classes]
        """
        class_counts = np.zeros(self.num_classes, dtype=np.int64)
        for sid in self.sample_ids:
            cluster_label = self.pseudo_labels[sid]
            class_idx = self.label2idx[cluster_label]
            class_counts[class_idx] += 1
        class_valid_flag = class_counts > 0
        valid_count = np.sum(class_valid_flag)
        
        class_weights = np.zeros(self.num_classes, dtype=np.float32)
        class_weights[class_valid_flag] = 1.0 / (class_counts[class_valid_flag])**0.5
        class_weights = class_weights / np.sum(class_weights) * valid_count  # 归一化
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def preprocess(self, img):
        """预处理图像，与ONNX模型保持一致"""
        # BGR to RGB
        img = img[:, :, ::-1]
        # Resize to 112x112
        img = cv2.resize(img, (112, 112))
        # Transpose to CHW
        img = np.transpose(img, axes=(2, 0, 1))
        # Normalize: (img / 255. - 0.5) / 0.5
        img = (img / 255.0 - 0.5) / 0.5
        return img.astype(np.float32)
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        face_id = self.sample_ids[index]
        # Parse episode_id and face_idx from face_id (format: "E01-1_0")
        episode_id = face_id.rsplit('-', 1)[0]
        
        # Load face image
        img_path = os.path.join(self.dataset_dir, episode_id, f'{face_id}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        # Preprocess image
        img_tensor = torch.from_numpy(self.preprocess(img))
        # Get pseudo-label
        cluster_label = self.pseudo_labels[face_id]
        class_idx = self.label2idx[cluster_label]
        
        return img_tensor, class_idx


class LengthAwareBatchSampler(Sampler):
    """
    自定义 BatchSampler，根据样本长度对数据进行分组，并支持每个 epoch 随机打乱 batch 顺序。
    注意:当使用 hidden features 时,不再需要按长度排序,因为不需要裁剪。
    """
    def __init__(self, dataset, batch_size, shuffle=True, use_hidfeat_flag=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_hidfeat_flag = use_hidfeat_flag
        self.indices = list(range(len(dataset)))
        if not self.use_hidfeat_flag:
            # 只有在使用原始语音特征时才按长度排序，要求数据集必须有长度信息
            assert hasattr(dataset, 'sample_lengths'), "Dataset must have sample_lengths attribute for length-aware batching when not using hidden features!"
            self.indices.sort(key=lambda idx: dataset.sample_lengths[dataset.sample_ids[idx]])  # 按长度排序

    def __iter__(self):
        if not self.use_hidfeat_flag:   # 按 batch_size 分组
            batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
            if self.shuffle:
                random.seed(torch.initial_seed())  # 使用全局随机数种子
                random.shuffle(batches)  # 打乱 batch 顺序
        else:
            if self.shuffle:
                random.seed(torch.initial_seed())  # 使用全局随机数种子
                random.shuffle(self.indices)  # 打乱 sample 顺序
            batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

class MLPClassifier(nn.Module):
    """
    Simple MLP classifier head for fine-tuning.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def collate_fn_hidden(batch):
    """
    Collate function for hidden features (don't need cropping).
    Args:
        batch: List of tuples (feat, label)
    Returns:
        feats: batch特征张量 (stacked)
        labels: batch标签张量
    """
    feats, labels = zip(*batch)
    # Stack features directly
    stacked_feats = torch.stack(feats)
    labels = torch.tensor(labels)
    return stacked_feats, labels    # of shape [batch_size, feat_dim], [batch_size]

def collate_fn(batch):
    """
    Collate function for original features, used for center cropping samples within a batch.
    Args:
        batch: List of tuples (feat, label, length)
    Returns:
        padded_feats: 裁剪后的特征张量
        labels: 标签张量
    """
    # 按样本长度排序（从大到小）
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    feats, labels, lengths = zip(*batch)

    # 计算裁剪后的目标长度（取 batch 内最短样本长度）
    target_length = min(lengths)

    # 对每个样本进行中心裁剪
    cropped_feats = []
    for feat in feats:
        if feat.shape[0] == target_length:
            cropped_feats.append(feat)
        else:
            start_idx = (feat.shape[0] - target_length) // 2
            end_idx = start_idx + target_length
            # 防止由于奇偶数导致裁剪长度不一致
            cropped_feats.append(feat[start_idx:end_idx])
    assert all(f.shape[0] == target_length for f in cropped_feats), "裁剪后长度不一致"

    # 将裁剪后的特征和标签打包成张量
    padded_feats = torch.stack(cropped_feats)
    labels = torch.tensor(labels)
    return padded_feats, labels # of shape [batch_size, target_length, feat_dim], [batch_size]

def unfrozen_model_layers(model, unfrozen_layers_num=0, print_mod_flag=False):
    """
    Unfreeze the top layers of the model.
    
    Args:
        model: The embedding model
        unfrozen_layers_num: Number of layers to unfreeze from the top. Not exact, since both leaf and non-leaf modules are counted.
    """
    # Get all modules
    all_modules = list(model.named_modules())
    if print_mod_flag:
        print(f"[INFO] Total layers in model: {len(all_modules)}")
        print(f"[INFO] All layers:")
        for idx, (name, module) in enumerate(all_modules):
            print(f"  [{idx}] {name}")
            print(f"    {module}")

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze top layers
    if unfrozen_layers_num > 0:
        for idx, (name, module) in enumerate(all_modules):
            if idx >= len(all_modules) - unfrozen_layers_num:
                for param in module.parameters():
                    param.requires_grad = True
                print(f"[INFO] Unfroze layer [{idx}] {name}")

def unfrozen_model_spk_modules(model, unfrozen_modules_num=1, print_mod_flag=False):
    """
    Unfreeze the top layers of the model at module level.
    
    Args:
        model: The embedding model
        unfrozen_modules_num: Number of modules to unfreeze from the top (based on xvector's named_children)
        print_mod_flag: Whether to print module information
    """
    # Get all named children of xvector (module level)
    xvector_modules = list(model.xvector.named_children())
    num_frozen_modules = len(xvector_modules) - unfrozen_modules_num
    
    if print_mod_flag:
        # Print all top-level modules in model
        print(f"[INFO] Total modules in model: {len(list(model.named_children()))}")
        print(f"[INFO] All top-level modules in model:")
        for idx, (name, module) in enumerate(model.named_children()):
            print(f"  [{idx}] {name}: {module.__class__.__name__}")
        
        # Print all xvector modules
        if unfrozen_modules_num > 0:
            print(f"[INFO] Total modules in xvector: {len(xvector_modules)}")
            print(f"[INFO] All xvector modules:")
            for idx, (name, module) in enumerate(xvector_modules):
                if idx < num_frozen_modules:
                    print(f"[INFO] Frozen module [{idx}] {name}")
                else:
                    print(f"[INFO] Unfrozen module [{idx}] {name}")
                print(f"    {module}")
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze top modules at module level
    if unfrozen_modules_num > 0:
        for idx, (name, module) in enumerate(xvector_modules):
            if idx >= num_frozen_modules:
                for param in module.parameters():
                    param.requires_grad = True
                print(f"[INFO] Unfroze module [{idx}] {name} in xvector")


def unfrozen_model_face_modules(model, print_mod_flag=False):
    """
    Unfreeze the output_layer of the face recognition model (IR_101).
    
    Args:
        model: The face recognition model (IR_101)
        print_mod_flag: Whether to print module information
    """
    # Get all named children of the model (module level)
    model_modules = list(model.named_children())
    
    if print_mod_flag:
        # Print all top-level modules in model
        print(f"[INFO] Total modules in face model: {len(model_modules)}")
        print(f"[INFO] All top-level modules in face model:")
        for idx, (name, module) in enumerate(model_modules):
            print(f"  [{idx}] {name}: {module.__class__.__name__}")
        
        # Print output_layer structure
        print(f"[INFO] output_layer will be unfrozen")
        if hasattr(model, 'output_layer'):
            print(f"[INFO] output_layer structure:")
            for sub_idx, (sub_name, sub_module) in enumerate(model.output_layer.named_children()):
                print(f"    [{sub_idx}] {sub_name}: {sub_module}")
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze output_layer
    if hasattr(model, 'output_layer'):
        for param in model.output_layer.parameters():
            param.requires_grad = True
        print(f"[INFO] Unfroze output_layer in face model")
    else:
        print(f"[WARNING] Model does not have output_layer attribute")


def inference_with_classifier(model, classifier, dataset, device, compute_uncertainty=False, batch_process_flag=True, batch_size=None, use_hidfeat=False, unfrozen_modules_num=1, potential_set=None):
    """
    Perform inference on dataset using model and classifier.
    Supports both speaker (audio) and face (vision) modalities.
    
    Args:
        model: Embedding model (speaker or face recognition model)
        classifier: Classifier head
        dataset: Dataset containing samples (PseudoLabelDataset or PseudoLabelFaceDataset)
        device: Torch device
        compute_uncertainty: Whether to compute uncertainty and potential labels
        batch_process_flag: Whether to use batch processing (for speaker model, only when use_hidfeat=true)
        batch_size: Batch size for inference (only used when use_hidfeat=True)
        use_hidfeat: Whether using hidden features for speaker model (enables batch inference)
        unfrozen_modules_num: Number of unfrozen modules (for forward_from with hidden features)
        potential_set: List of potential class indices to consider for uncertainty computation (optional)
    
    Returns:
        preds_dic: {sample_id: predicted_label}
        embeddings_dic: {sample_id: embedding_array}
        uncertainty_dic: {sample_id: uncertainty_score} (if compute_uncertainty=True)
        potential_list_dic: {sample_id: [top2_label_list]} (if compute_uncertainty=True)
    """
    model.eval()
    classifier.eval()
    
    preds_dic, embeddings_dic = {}, {}
    if compute_uncertainty:
        uncertainty_dic, potential_list_dic = {}, {}
        if potential_set is not None:
            # Filter probabilities to only include potential_set indices
            potential_indices = [dataset.label2idx[label] for label in potential_set if label in dataset.label2idx]
    else:
        uncertainty_dic, potential_list_dic = None, None
    
    with torch.no_grad():
        if batch_process_flag:  # Batch processing: Speaker model with hidden features or Face model
            collate_fn_used = collate_fn_hidden if use_hidfeat else None
            inference_loader = DataLoader(dataset, batch_size=batch_size, 
                                          sampler=SequentialSampler(dataset),
                                          num_workers=4, pin_memory=True, collate_fn=collate_fn_used)
            
            sample_idx = 0
            for batch_feat, _ in inference_loader:
                batch_feat = batch_feat.to(device)  # [B, ...]
                batch_size_actual = batch_feat.size(0)
                
                # Forward pass
                if use_hidfeat:
                    batch_emb = model.forward_from(batch_feat, unfrozen_modules_num)
                else:
                    batch_emb = model(batch_feat)  # [B, emb_dim]
                batch_logits = classifier(batch_emb)
                batch_probs = torch.softmax(batch_logits, dim=1)  # [B, num_classes]
                batch_pred_labels = torch.argmax(batch_probs, dim=1)  # [B]
                
                # Map back to sample IDs and store results
                for i in range(batch_size_actual):
                    sid = dataset.sample_ids[sample_idx]
                    pred_label = int(dataset.idx2label[batch_pred_labels[i].item()])
                    preds_dic[sid] = pred_label
                    embeddings_dic[sid] = batch_emb[i].squeeze(0).cpu().numpy()
                    
                    if compute_uncertainty:
                        probs = batch_probs[i]
                        top2_probs, top2_indices = torch.topk(probs, 2)
                        uncertainty = (top2_probs[0] - top2_probs[1]).item()
                        uncertainty_dic[sid] = float(uncertainty)
                        if potential_set is not None:   # Filter top2 within potential_set
                            _, top2_indices_potential = torch.topk(probs[potential_indices], 2)
                            top2_indices = [potential_indices[idx] for idx in top2_indices_potential.cpu().numpy()]
                        else:
                            top2_indices = top2_indices.cpu().numpy().tolist()
                        top2_labels = [int(dataset.idx2label[idx]) for idx in top2_indices]
                        potential_list_dic[sid] = top2_labels
                    sample_idx += 1
        
        else:  # Single sample processing: Speaker model without hidden features
            for sample_id in dataset.sample_ids:
                # Get sample from dataset
                if hasattr(dataset, 'subseg_feat_dic'):  # Speaker dataset with original features
                    feat = dataset.subseg_feat_dic[sample_id]
                    feat = feat.unsqueeze(0).to(device)  # [1, T, D]
                    emb = model(feat)  # [1, emb_dim]
                else:  # Face dataset
                    idx = dataset.sample_ids.index(sample_id)
                    img_tensor, _ = dataset[idx]
                    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
                    emb = model(img_tensor)  # [1, emb_dim]
                
                # Get predictions
                logits = classifier(emb)
                probs = torch.softmax(logits, dim=1)[0]  # [num_classes]
                pred_label = torch.argmax(probs).item()
                pred_label = int(dataset.idx2label[pred_label])
                
                preds_dic[sample_id] = pred_label
                embeddings_dic[sample_id] = emb.squeeze(0).cpu().numpy()
                
                if compute_uncertainty:
                    top2_probs, top2_indices = torch.topk(probs, 2)
                    uncertainty = (top2_probs[0] - top2_probs[1]).item()
                    uncertainty_dic[sample_id] = float(uncertainty)
                    if potential_set is not None:   # Filter top2 within potential_set
                        _, top2_indices_potential = torch.topk(probs[potential_indices], 2)
                        top2_indices = [potential_indices[idx] for idx in top2_indices_potential.cpu().numpy()]
                    else:
                        top2_indices = top2_indices.cpu().numpy().tolist()
                    top2_labels = [int(dataset.idx2label[idx]) for idx in top2_indices]
                    potential_list_dic[sid] = top2_labels

    if compute_uncertainty and potential_set is not None:
        # Gather all values from potential_list_dic, make them unique, and assert they are a subset of potential_set
        potential_list_labels = set(label for labels in potential_list_dic.values() for label in labels)
        # print(f"[INFO] Potential set: {potential_set}")
        # print(f"[INFO] Unique labels in potential_list_dic: {potential_list_labels}")
        assert potential_list_labels.issubset(potential_set), "Potential list labels are not a subset of the potential set!"
    return preds_dic, embeddings_dic, uncertainty_dic, potential_list_dic


def train_one_epoch(train_loader, model, classifier, optimizer, epoch, logger, device, use_hidfeat_flag=False, unfrozen_modules_num=1):
    """
    Train for one epoch without gradient accumulation.
    
    Args:
        use_hidfeat_flag: If True, input is hidden features and use forward_from
        unfrozen_modules_num: Number of unfrozen modules from top. always be 1 when use_hidfeat_flag is True, since the output of former modules have different dimension.(when use unfrozen_model_layers, unfrozen_layers_num needn't to be known here)
    """
    # criterion = nn.CrossEntropyLoss()
    class_weights = train_loader.dataset.class_weights.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_stats = AverageMeters()
    train_stats.add('Time', ':6.3f')
    train_stats.add('Loss', ':.4e')
    train_stats.add('Acc pseudo', ':6.2f')
    train_stats.add('Lr', ':.3e')
    
    progress = ProgressMeter(
        len(train_loader),
        train_stats,
        prefix="Epoch: [{}]".format(epoch)
    )
    
    model.train()
    classifier.train()
    
    end = time.time()
    
    for i, (feat, label) in enumerate(train_loader):
        if not use_hidfeat_flag:    # batch normalization cannot handle bs=1
            if feat.dim() == 2 or feat.size(0) == 1:
                continue
        feat = feat.to(device)
        label = label.to(device)
        
        # Use forward_from if using hidden features, otherwise full forward
        if use_hidfeat_flag:
            embedding = model.forward_from(feat, unfrozen_modules_num)
        else:
            embedding = model(feat)
        output = classifier(embedding)
        loss = criterion(output, label)
        acc = accuracy(output, label)
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        
        # Record statistics
        train_stats.update('Time', time.time() - end)
        train_stats.update('Loss', loss.item(), feat.size(0))
        train_stats.update('Acc pseudo', acc.item(), feat.size(0))
        train_stats.update('Lr', optimizer.param_groups[0]["lr"])
        
        if i % 50 == 0:
            logger.info(progress.display(i))
        
        end = time.time()
    
    return {
        'loss': train_stats.avg('Loss'),
        'acc': train_stats.avg('Acc pseudo'),
        'lr': train_stats.val('Lr')
    }


def extract_embeddings_with_model(speaker_model_id, speaker_model_path, conf_file, subseg_json, audio_embs_out_dir, use_gpu=False, gpu=None):
    """
    通过命令行调用 extract_diar_embeddings.py，提取所有语音的 embedding 并保存到指定目录。

    Args:
        speaker_model_id: 说话人模型ID（如 iic/speech_campplus_sv_zh-cn_3dspeaker_16k）
        speaker_model_path: 微调后的说话人模型路径
        subseg_json: 子片段信息json
        audio_embs_out_dir: 输出embedding目录
        use_gpu: 是否使用GPU（bool）
        gpu: GPU id 列表（如 [0] 或 [0,1]）
    """

    cmd = [
        sys.executable,
        'local/extract_diar_embeddings.py',
        '--model_id', speaker_model_id,
        '--conf', conf_file,
        '--subseg_json', subseg_json,
        '--embs_out', audio_embs_out_dir,
    ]
    if speaker_model_path is not None:
        cmd.extend(['--pretrained_model', speaker_model_path])
    if use_gpu:
        cmd.append('--use_gpu')
        if gpu is not None:
            if isinstance(gpu, (list, tuple)):
                gpu_ids = [str(g) for g in gpu if not isinstance(g, bool)]
                if gpu_ids:
                    cmd.extend(['--gpu'] + gpu_ids)
            # 单个 int 或 str（但排除 bool）直接添加
            elif isinstance(gpu, (int, str)) and not isinstance(gpu, bool):
                cmd.extend(['--gpu', str(gpu)])

    print(f"[INFO] Running embedding extraction with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"[WARNING] Stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Embedding extraction failed with error: {e}")
        print(f"[ERROR] Stdout: {e.stdout}")
        print(f"[ERROR] Stderr: {e.stderr}")
        raise e

def compute_acc_from_dic(pred_dic, ref_dic):
    correct_num = sum(list(map(lambda k: 1 if pred_dic[k] == ref_dic[k] else 0, ref_dic.keys())))
    ref_total = len(ref_dic)
    return correct_num / ref_total if ref_total > 0 else 0

def cmd_compute_acc(result_dir, anno_file, mode, modal='speaker'):
    """
    Call compute_acc_spk.py or compute_acc_face.py to compute accuracy.
    
    Args:
        result_dir: Directory containing result json files
        anno_file: Path to annotation xlsx file
        mode: Mode for accuracy computation ('valid', 'test' or 'all')
        modal: Modal type ('speaker' or 'face')
    
    Returns:
        accuracy: Overall speaker or face recognition accuracy
    """
    # Select script based on modal
    if modal == 'speaker':
        script = 'local/compute_acc_spk.py'
    elif modal == 'face':
        script = 'local/compute_acc_face.py'
    else:
        raise ValueError(f"Unsupported modal: {modal}")
    
    # Run accuracy computation script
    cmd = [
        sys.executable,
        script,
        '--result_dir', result_dir,
        '--ref_xlsx', anno_file,
        '--mode', mode
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Error computing accuracy: {e}")
        raise e

def get_acc(acc_file):
    """
    Parse accuracy from the given accuracy file.
    
    Args:
        acc_file: Path to accuracy file
    Returns:
        accuracy: Overall speaker recognition accuracy
    """
    try:
        with open(acc_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('overall_accuracy'):
                    acc = float(line.split(':')[1].strip())
                    return acc
    except Exception as e:
        print(f"[WARNING] Error parsing accuracy file: {e}")
        raise e

def compute_acc_from_anno(result_dir, anno_file, mode='all', modal='speaker'):
    """
    Compute speaker or face recognition accuracy from pseudo labels.
    
    Args:
        result_dir: Directory containing result json files
        anno_file: Path to annotation xlsx file
        mode: Mode for accuracy computation ('valid', 'test' or 'all')
        modal: Modal type ('speaker' or 'face')
    
    Returns:
        accuracy: Overall recognition accuracy
    """
    # Determine json file pattern based on modal
    if modal == 'speaker':
        json_pattern = 'pseudo_labels_audio'
    elif modal == 'face':
        json_pattern = 'pseudo_labels_faces_mid_frame'
    else:
        raise ValueError(f"Unsupported modal: {modal}")
    
    # check whether result_dir contains pseudo label json files
    pseudo_label_files = [f for f in os.listdir(result_dir) if f.endswith('.json') and json_pattern in f]
    assert len(pseudo_label_files) <= 1, f"Multiple pseudo-label files found in {result_dir}: {pseudo_label_files}"
    if len(pseudo_label_files) == 0:
        pseudo_labels_all_dir = os.path.join(result_dir, "pseudo_labels_audio_unreliable_pp")
        assert os.path.exists(pseudo_labels_all_dir), f"No pseudo-label files found in {result_dir} or its subdirectories!"
        pseudo_label_files = [f for f in os.listdir(pseudo_labels_all_dir) if f.endswith('.json') and 'pseudo_labels_audio' in f]
        if len(pseudo_label_files) == 0:
            raise AssertionError(f"No pseudo-label files found in {pseudo_labels_all_dir}!")
        elif len(pseudo_label_files) == 1:
            best_idx = 0
        else:
            # Find the pseudo label file with highest valid accuracy
            cmd_compute_acc(pseudo_labels_all_dir, anno_file, mode='valid', modal=modal)
            pseudo_label_files_acc_val = [get_acc(os.path.join(pseudo_labels_all_dir, f.replace('.json', '_accuracy(valid).txt'))) for f in pseudo_label_files]
            best_idx = int(np.argmax(np.array(pseudo_label_files_acc_val)))
            print(f"[INFO] Selected pseudo-label file {pseudo_label_files[best_idx]} with highest valid accuracy {pseudo_label_files_acc_val[best_idx]:.4f}")
            cmd_compute_acc(pseudo_labels_all_dir, anno_file, mode='test', modal=modal) # used for logging purpose
        
        # Move the pseudo label file in pseudo_labels_all_dir with highest valid acc to result_dir
        src_path = os.path.join(pseudo_labels_all_dir, pseudo_label_files[best_idx])
        dst_path = os.path.join(result_dir, pseudo_label_files[best_idx])
        shutil.move(src_path, dst_path)
    
    # Run accuracy computation script
    cmd_compute_acc(result_dir, anno_file, mode, modal)

    # Parse accuracy from the output file
    if mode == 'all':
        acc_files = [f for f in os.listdir(result_dir) if f.endswith(f'_accuracy.txt') and json_pattern in f]
    elif mode in ['valid', 'test']:
        acc_files = [f for f in os.listdir(result_dir) if f.endswith(f'_accuracy({mode}).txt') and json_pattern in f]
    else:
        raise ValueError(f"Unsupported mode {mode} for accuracy computation!")
    assert len(acc_files) > 0, f"No accuracy file found in {result_dir}"
    assert len(acc_files) == 1, f"Multiple accuracy files found in {result_dir}: {acc_files}"
    acc = get_acc(os.path.join(result_dir, acc_files[0]))
    return acc


def run_clustering_and_evaluation(conf_file, cluster_type, wavs, audio_embs_dir, visual_embs_dir, result_dir, hmm_flag, fix_mf_flag, hmm_visual_info_type, unreliable_pp, speaker_anno_file, face_anno_file=None, hmm_model_path=None, from_preds=False, mode='all'):
    """
    Run clustering with HMM correction and evaluate accuracy.
    
    Args:
        conf_file: Configuration file for clustering
        cluster_type: Type of clustering ('audio_only' or 'audio_vision')
        wavs: Wav list file
        audio_embs_dir: Directory of audio embeddings
        visual_embs_dir: Directory of visual embeddings
        result_dir: Directory to save pseudo labels
        hmm_flag: Whether to use HMM smoothing
        fix_mf_flag: Whether to fix key frame visual cluster labels during HMM smoothing
        hmm_visual_info_type: Visual information type for HMM
        unreliable_pp: Percentage of unreliable segments to be smoothed
        speaker_anno_file: Speaker annotation xlsx file
        face_anno_file: Face annotation xlsx file (optional)
        hmm_model_path: Path to HMM model (optional)
        from_preds: Whether to use local predictions from classifier model instead of clustering (optional)
        mode: Mode for accuracy computation ('valid', 'test' or 'all')
    
    Returns:
        speaker_acc: Speaker recognition accuracy
        face_acc: Face recognition accuracy (only if cluster_type=='audio_vision' and face_anno_file is provided)
    """
    # Prepare command to call cluster_and_postprocess.py
    cmd = [
        sys.executable,
        'local/cluster_and_postprocess.py',
        '--conf', conf_file,
        '--cluster_type', cluster_type,
        '--wavs', wavs,
        '--audio_embs_dir', audio_embs_dir,
        '--result_dir', result_dir
    ]
    if hmm_flag:
        cmd.append('--use_hmm_smoothing')
    
    # Add visual embeddings parameters if using audio-vision clustering
    if cluster_type == 'audio_vision':
        cmd.extend(['--visual_embs_dir', visual_embs_dir])
        if fix_mf_flag:
            cmd.append('--fix_mf')
        cmd.extend(['--hmm_visual_info_type', hmm_visual_info_type])
        cmd.extend(['--unreliable_pp', str(unreliable_pp)])
    
    if hmm_model_path is not None:
        cmd.extend(['--hmm_model_path', hmm_model_path])
    if from_preds:
        cmd.append('--from_preds')
    # Run clustering
    print(f"[INFO] Running clustering with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"[WARNING] Stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Clustering failed with error: {e}")
        print(f"[ERROR] Stdout: {e.stdout}")
        print(f"[ERROR] Stderr: {e.stderr}")
        raise e
    
    # Compute speaker accuracy
    assert os.path.exists(speaker_anno_file), f"Speaker annotation file {speaker_anno_file} does not exist!"
    speaker_acc = compute_acc_from_anno(result_dir, speaker_anno_file, mode, modal='speaker')
    
    # Compute face accuracy if applicable
    face_acc = None
    if cluster_type == 'audio_vision' and face_anno_file is not None:
        if os.path.exists(face_anno_file):
            face_acc = compute_acc_from_anno(result_dir, face_anno_file, mode, modal='face')
        else:
            print(f"[WARNING] Face annotation file {face_anno_file} does not exist, skipping face accuracy computation.")
    
    return speaker_acc, face_acc


def main():
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    torch.manual_seed(args.seed)  # 设置 PyTorch 的随机数种子
    if args.use_gpu:
        torch.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 的随机数种子
    
    # Setup distributed training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        world_size = 1
        if args.use_gpu and len(args.gpu) > 1:
            dist.init_process_group(backend='nccl')
            rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
    
    # Set device
    if args.use_gpu:
        gpu_id = int(args.gpu[rank % len(args.gpu)])
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    # Create directories
    ## Create directory for self-supervised fine-tuning
    finetune_dir = os.path.join(args.result_dir, 'self_supervised')
    os.makedirs(finetune_dir, exist_ok=True)
    ## Create dictory for current experiment
    existing_exp_dirs = [d for d in os.listdir(finetune_dir) if os.path.isdir(os.path.join(finetune_dir, d)) and d.startswith("exp") and d[3:].isdigit()]
    if len(existing_exp_dirs) > 0:
        existing_exp_dirs.sort(key=lambda x: int(x[3:]))  # Sort directories by their numeric suffix
        exp_name = f"exp{int(existing_exp_dirs[-1][3:]) + 1}"
    else:
        exp_name = "exp0"
    exp_dir = os.path.join(finetune_dir, exp_name)
    assert not os.path.exists(exp_dir), f"Experiment directory {exp_dir} already exists!"   # NOTE: 一次性跑多个实验时，需要等待前一个实验目录创建完成
    os.makedirs(exp_dir, exist_ok=True)
    
    # Setup logger
    logger = get_logger(os.path.join(exp_dir, 'self_supervised_train.log'))
    logger.info(f"Starting self-supervised fine-tuning pipeline")
    logger.info(f"Device: {device}")

    # Save args to a dictionary and write to a JSON file
    args_json_path = os.path.join(exp_dir, 'args.json')
    with open(args_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Saved arguments to {args_json_path}")
    
    # ============================
    # Initial clustering
    # ============================
    logger.info("="*20)
    logger.info("Initialization: Initial clustering with HMM correction")
    logger.info("="*20)
    
    initial_dir = os.path.join(exp_dir, 'initial')
    pseudo_label_dir = os.path.join(initial_dir, 'pseudo_label')
    os.makedirs(pseudo_label_dir, exist_ok=True)
    # Check if pseudo_label_dir is an empty folder
    if os.path.exists(os.path.join(finetune_dir, 'initial')):
        logger.info(f"Skipping initial clustering, using existing results.")
        shutil.copytree(os.path.join(finetune_dir, 'initial'), initial_dir, dirs_exist_ok=True)
        initial_acc_spk = compute_acc_from_anno(pseudo_label_dir, args.speaker_anno_file, mode='all', modal='speaker')
        if args.cluster_type == 'audio_vision' and os.path.exists(args.face_anno_file):
            initial_acc_face = compute_acc_from_anno(pseudo_label_dir, args.face_anno_file, mode='all', modal='face')
        else:
            initial_acc_face = None
    else:
        initial_acc_spk, initial_acc_face = run_clustering_and_evaluation(
            args.conf,
            args.cluster_type,
            args.wavs,
            args.audio_embs_dir,    # 预先提取的音频embedding所在目录
            args.visual_embs_dir,
            pseudo_label_dir,
            args.use_hmm_smoothing,
            args.fix_mf,
            args.hmm_visual_info_type,
            args.unreliable_pp,
            args.speaker_anno_file,
            args.face_anno_file,
            hmm_model_path=None,
            from_preds=False,
            mode='all'
        )
        shutil.copytree(initial_dir, os.path.join(finetune_dir, 'initial'), dirs_exist_ok=True)
        logger.info(f"Saved initial clustering results to {initial_dir}")
    
    logger.info(f"Initial: speaker_acc={initial_acc_spk:.4f}")
    if initial_acc_face is not None:
        logger.info(f"Initial: face_acc={initial_acc_face:.4f}")
    # Accuracy history
    acc_history_spk = [{'round': 'Initial', 'acc': initial_acc_spk}]
    acc_history_face = [{'round': 'Initial', 'acc': initial_acc_face}] if initial_acc_face is not None else None

    # Load unreliable segment IDs and initial cluster results
    with open(os.path.join(pseudo_label_dir, 'useful_var_dic.pkl'), 'rb') as f:
        useful_var_dic = pickle.load(f)
    audio_seg_ids = useful_var_dic['audio_seg_ids']
    alabels_unreliable_metrics_init = useful_var_dic['alabels_unreliable_metrics_init']
    idxs_unreliable = np.argsort(alabels_unreliable_metrics_init)[:int(args.unreliable_pp / 100 * len(audio_seg_ids))]
    audio_seg_ids_unreliable = audio_seg_ids[idxs_unreliable]

    # define pseudo_valid_label_dic for speaker and save
    with open(os.path.join(pseudo_label_dir, 'cluster_results_vision_vad_processed_for_HMM_nested_X_uniq.json'), 'r', encoding='utf-8') as f:
        vad_cluster_results = json.load(f)
    with open(os.path.join(pseudo_label_dir, 'cluster_results_audio_processed_for_HMM_nested_X.json'), 'r', encoding='utf-8') as f:
        audio_obs_init_results = json.load(f)
    pseudo_valid_label_dic_spk = {k: vad_cluster_results[k] for k in vad_cluster_results if vad_cluster_results[k] == audio_obs_init_results[k]}
    with open(os.path.join(pseudo_label_dir, 'pseudo_valid_labels_speaker.json'), 'w', encoding='utf-8') as f:
        json.dump(pseudo_valid_label_dic_spk, f, indent=2)

    # ============================
    # Iterative fine-tuning
    # ============================
    for round in range(args.max_rounds):
        logger.info("="*20)
        logger.info(f"Round {round}: Fine-tuning iteration")
        logger.info("="*20)
        
        round_dir = os.path.join(exp_dir, f'round{round}')
        os.makedirs(round_dir, exist_ok=True)
        round_pseudo_label_dir = os.path.join(round_dir, 'pseudo_label')
        os.makedirs(round_pseudo_label_dir, exist_ok=True)

        # ============================
        # Part 1: Fine-tune model
        # ============================
        logger.info(f"Round {round} Part 1: Fine-tuning embedding model")
        
        # Determine which pseudo labels to use
        pseudo_label_files = [f for f in os.listdir(pseudo_label_dir) if f.endswith('.json') and 'pseudo_labels_audio' in f]
        assert len(pseudo_label_files) > 0, f"No pseudo-label file found in {pseudo_label_dir}"
        assert len(pseudo_label_files) == 1, f"Multiple pseudo-label files found in {pseudo_label_dir}: {pseudo_label_files}"
        pseudo_label_file = os.path.join(pseudo_label_dir, pseudo_label_files[0])
        logger.info(f"Using pseudo-label file: {pseudo_label_file}")

        # Load model
        if round == 0:
            # ============ Speaker Model ============
            # get objects of feature extractor and embedding model
            conf_model = update_conf(yaml_config_loader(args.conf), args.speaker_model_id, args.speaker_pretrained_model, rank)
            config_model = Config(conf_model)
            feature_extractor = build('feature_extractor', config_model)
            embedding_model = build('embedding_model', config_model)
            
            # get output embedding dimension of the embedding model
            embedding_dim = conf_model['embedding_model']['args']['embedding_size']
            logger.info(f"Speaker embedding dimension: {embedding_dim}")

            # load pretrained model
            pretrained_state = torch.load(config_model.pretrained_model, map_location='cpu')
            embedding_model.load_state_dict(pretrained_state)
            embedding_model.to(device)
            # Check if the device is GPU
            if device.type == 'cuda':
                print(f"[INFO]: Speaker model using GPU: {torch.cuda.get_device_name(device)}")
            else:
                print("[INFO]: Speaker model using CPU")

            # Save config_model as JSON
            config_model_json_path = os.path.join(round_dir, 'config_model_speaker.json')
            with open(config_model_json_path, 'w') as f:
              json.dump(conf_model, f, indent=4)  # Assuming conf_model is serializable
            logger.info(f"Saved speaker config_model to {config_model_json_path}")
            
            # # Copy pretrained model to local round dir for record
            # torch.save(embedding_model.state_dict(), os.path.join(initial_dir, 'pretrained_model_speaker.pth'))
            
            # ============ Face Model ============
            if args.cluster_type == 'audio_vision':
                # Initialize face embedding model
                face_embedding_model = IR_101(input_size=(112, 112))
                face_embedding_dim = 512  # IR_101 output dimension
                logger.info(f"Face embedding dimension: {face_embedding_dim}")
                
                # Load pretrained face model
                if os.path.exists(args.face_pretrained_model):
                    logger.info(f"Loading face pretrained model from {args.face_pretrained_model}")
                    checkpoint = torch.load(args.face_pretrained_model, map_location='cpu')
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    face_embedding_model.load_state_dict(state_dict, strict=True)
                    logger.info("Face model weights loaded successfully")
                else:
                    raise FileNotFoundError(f"Face pretrained model not found: {args.face_pretrained_model}")
                
                face_embedding_model.to(device)
                if device.type == 'cuda':
                    print(f"[INFO]: Face model using GPU: {torch.cuda.get_device_name(device)}")
                else:
                    print("[INFO]: Face model using CPU")
                
                # # Copy pretrained face model to local round dir for record
                # torch.save(face_embedding_model.state_dict(), os.path.join(initial_dir, 'pretrained_model_face.pth'))

        # Create speaker dataset and dataloader
        if round == 0:
            spk_update_class_flag = True
            train_dataset = PseudoLabelDataset(args.subseg_json, pseudo_label_file, feature_extractor)
            crt_collate_fn = collate_fn_hidden if args.use_hidfeat else collate_fn
            subseg_hidfeat_dic = None
            if args.use_hidfeat:    # Define subseg_hidfeat_dic at round 0
                logger.info(f"Extracting hidden features for all speaker samples, which are outputs of StatsPool layer...")
                logger.info(f"args.unfrozen_layers_num={args.unfrozen_layers_num} is not used when extracting hidden features.")
                subseg_hidfeat_dic = {}
                embedding_model.eval()
                with torch.no_grad():
                    for sid in train_dataset.sample_ids:
                        feat = train_dataset.subseg_feat_dic[sid].unsqueeze(0).to(device)  # [1, T, F]
                        hidden_feat = embedding_model.forward_until(feat, 1)
                        subseg_hidfeat_dic[sid] = hidden_feat.squeeze(0).detach().cpu()  # [hid_dim]
                logger.info(f"Hidden features extracted for {len(subseg_hidfeat_dic)} speaker samples")
                train_dataset = PseudoLabelDataset(args.subseg_json, pseudo_label_file, feature_extractor, args.use_hidfeat, subseg_hidfeat_dic)
        else:   # Re-create dataset with new pseudo_label_file
            spk_update_class_flag = train_dataset.reset_labels(pseudo_label_file)
        
        if len(train_dataset) == 0:
            logger.error("No valid speaker training samples found!")
            break
        batch_sampler = LengthAwareBatchSampler(train_dataset, batch_size=args.finetune_batch_size, shuffle=True, use_hidfeat_flag=args.use_hidfeat)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, collate_fn=crt_collate_fn)
        
        # Create face dataset and dataloader (if using audio_vision)
        if args.cluster_type == 'audio_vision':
            # Find face pseudo label file
            face_pseudo_label_files = [f for f in os.listdir(pseudo_label_dir) if f.endswith('.json') and 'pseudo_labels_faces_mid_frame' in f]
            if len(face_pseudo_label_files) > 0:
                assert len(face_pseudo_label_files) == 1, f"Multiple face pseudo-label files found in {pseudo_label_dir}: {face_pseudo_label_files}"
                face_pseudo_label_file = os.path.join(pseudo_label_dir, face_pseudo_label_files[0])
                logger.info(f"Using face pseudo-label file: {face_pseudo_label_file}")
                
                # define pseudo_valid_label_dic for face and save
                with open(pseudo_label_file, 'r', encoding='utf-8') as f:
                    pseudo_label_audio = json.load(f)
                if round == 0:
                    with open(face_pseudo_label_file, 'r', encoding='utf-8') as f:
                        pseudo_label_face = json.load(f)
                    # 将face ids按audio_seg_id分组
                    face_ids_dic = {}
                    for face_id in pseudo_label_face:
                        audio_seg_id = face_id.rsplit('_', 1)[0]
                        if audio_seg_id not in face_ids_dic:
                            face_ids_dic[audio_seg_id] = []
                        face_ids_dic[audio_seg_id].append(face_id)
                    # 根据speaker信息筛选face ids
                    face_ids_filtered_list_spk = []
                    for key in face_ids_dic:
                        face_ids_spk = []
                        for face_id in face_ids_dic[key]:
                            if pseudo_label_audio[key] == pseudo_label_face[face_id]:
                                face_ids_spk.append(face_id)
                        if len(face_ids_spk) == 1:
                            face_ids_filtered_list_spk.extend(face_ids_spk)
                    pseudo_valid_label_dic_face = {k: pseudo_label_face[k] for k in face_ids_filtered_list_spk}
                    with open(os.path.join(pseudo_label_dir, 'pseudo_valid_labels_face.json'), 'w', encoding='utf-8') as f:
                        json.dump(pseudo_valid_label_dic_face, f, indent=2)
                
                
                # Create face dataset
                if round == 0:
                    face_update_class_flag = True
                    face_train_dataset = PseudoLabelFaceDataset(face_pseudo_label_file, args.midframe_face_dir)
                else:
                    face_update_class_flag = face_train_dataset.reset_labels(face_pseudo_label_file)
                if len(face_train_dataset) == 0:
                    logger.error("No valid face training samples found!")
                    face_train_loader = None
                else:
                    face_train_loader = DataLoader(face_train_dataset, batch_size=args.finetune_batch_size, 
                                                   shuffle=True, num_workers=4, pin_memory=True)
                    logger.info(f"Created face dataloader with {len(face_train_dataset)} samples")
            else:
                logger.warning(f"No face pseudo-label file found in {pseudo_label_dir}, skipping face training")
                face_train_loader = None
        else:
            face_train_loader = None

        # ============ Speaker Model: Freeze and Create Classifier ============
        # Freeze embedding model initially
        for param in embedding_model.parameters():
            param.requires_grad = False
        
        # Create speaker classifier and warm up
        if spk_update_class_flag:
            # Create speaker classifier
            classifier = MLPClassifier(input_dim=embedding_dim, num_classes=train_dataset.num_classes).to(device)
        
            # Warmup: train classifier only when round == 0
            ## Optimizer for warmup (only classifier).
            optimizer_spk = torch.optim.Adam(classifier.parameters(), lr=1e-3)
            ## Warmup training
            logger.info(f"Warmup training speaker classifier for {args.warmup_epochs_num} epochs...")
            for warmup_epoch in range(args.warmup_epochs_num):
                torch.manual_seed(args.seed + warmup_epoch)
                if args.use_gpu:
                    torch.cuda.manual_seed_all(args.seed + warmup_epoch)
                train_stats_spk = train_one_epoch(train_loader,  embedding_model, classifier, optimizer_spk, 
                                                warmup_epoch, logger, device, args.use_hidfeat)
                logger.info(f"Round {round}, Speaker Warmup Epoch {warmup_epoch}: loss={train_stats_spk['loss']:.4f}, acc(pseudo)={train_stats_spk['acc']:.2f}%")
            optimizer_spk.zero_grad()
        
        # ============ Face Model: Freeze and Create Classifier ============
        if args.cluster_type == 'audio_vision' and face_train_loader is not None:
            # Freeze face embedding model initially
            for param in face_embedding_model.parameters():
                param.requires_grad = False
            
            # Create face classifier and warm up
            if face_update_class_flag:
                # Create face classifier
                face_classifier = MLPClassifier(input_dim=face_embedding_dim, num_classes=face_train_dataset.num_classes).to(device)
                
                # Warmup: train face classifier only when round == 0
                ## Optimizer for warmup (only face classifier)
                optimizer_face = torch.optim.Adam(face_classifier.parameters(), lr=1e-3)
                ## Warmup training
                logger.info(f"Warmup training face classifier for {args.warmup_epochs_num} epochs...")
                for warmup_epoch in range(args.warmup_epochs_num):
                    torch.manual_seed(args.seed + warmup_epoch)
                    if args.use_gpu:
                        torch.cuda.manual_seed_all(args.seed + warmup_epoch)
                    train_stats_face = train_one_epoch(face_train_loader,  face_embedding_model, face_classifier, optimizer_face, 
                                                    warmup_epoch, logger, device, use_hidfeat_flag=False)
                    logger.info(f"Round {round}, Face Warmup Epoch {warmup_epoch}: loss={train_stats_face['loss']:.4f}, acc(pseudo)={train_stats_face['acc']:.2f}%")
                optimizer_face.zero_grad()

        # ============================
        # Speaker Model Fine-tuning
        # ============================
        # Unfreeze last few layers in speaker embedding model
        if args.use_hidfeat:
            unfrozen_model_spk_modules(embedding_model, unfrozen_modules_num=1, print_mod_flag=(round == 0))
        else:
            unfrozen_model_layers(embedding_model, args.unfrozen_layers_num, print_mod_flag=(round == 0))

        # Optimizer for speaker fine-tuning (both embedding and classifier)
        optimizer_spk = torch.optim.Adam(
            list(embedding_model.parameters()) + list(classifier.parameters()),
            lr=args.finetune_lr
        )
        
        # Fine-tuning
        round_model_save_path_spk = os.path.join(round_dir, 'finetuned_model_speaker.pth')
        round_classifier_save_path_spk = os.path.join(round_dir, 'finetuned_classifier_speaker.pth')
        best_acc_valid_e_pseudo_spk, best_epoch_spk, patience_counter_epoch_spk = 0.0, 0, 0
        prev_acc_valid_e_pseudo_spk = 0.0
        if args.from_preds:
            best_preds_dic_spk, best_uncertainty_dic_spk, best_potential_list_dic_spk = {}, {}, {}
        logger.info(f"Fine-tuning speaker embedding model for {args.max_finetune_epochs} epochs...")
        for ft_epoch in range(args.max_finetune_epochs):
            torch.manual_seed(args.seed + ft_epoch)
            if args.use_gpu:
                torch.cuda.manual_seed_all(args.seed + ft_epoch)
            train_stats_spk = train_one_epoch(train_loader,  embedding_model, classifier, optimizer_spk, 
                                          ft_epoch, logger, device, args.use_hidfeat)
            logger.info(f"Round {round}, Fine-tune Epoch {ft_epoch}: loss={train_stats_spk['loss']:.4f}, acc(pseudo)={train_stats_spk['acc']:.2f}%")
            
            # === 每个epoch后，计算所有样本的分类标签、概率和不确定度 ===
            # Use unified inference function
            preds_dic, embeddings_dic, uncertainty_dic, potential_list_dic = inference_with_classifier(
                model=embedding_model,
                classifier=classifier,
                dataset=train_dataset,
                device=device,
                compute_uncertainty=args.from_preds,
                batch_process_flag=args.use_hidfeat,
                batch_size=args.finetune_batch_size * 2,
                use_hidfeat=args.use_hidfeat,
                unfrozen_modules_num=1,
            )


            # 计算在验证集上的准确率
            epoch_dir = os.path.join(round_dir, f'ft_epoch_{ft_epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            with open(os.path.join(epoch_dir, f'pseudo_labels_audio_pred.json'), 'w') as f:
                json.dump(preds_dic, f, indent=2)
            crt_acc_valid_e_pseudo_spk = compute_acc_from_dic(preds_dic, pseudo_valid_label_dic_spk)
            crt_acc_e = compute_acc_from_anno(epoch_dir, args.speaker_anno_file, mode='all', modal='speaker')
            logger.info(f"Round {round}, Fine-tune Epoch {ft_epoch}: acc(valid_pseudo): {crt_acc_valid_e_pseudo_spk:.4f}, acc: {crt_acc_e:.4f}")
            if crt_acc_valid_e_pseudo_spk > best_acc_valid_e_pseudo_spk:  # epoch0 must be better
                best_acc_valid_e_pseudo_spk, best_epoch_spk  = crt_acc_valid_e_pseudo_spk, ft_epoch
                patience_counter_epoch_spk = 0
                # Save best model of this round, and preds, uncertainty, potential_list dicts
                if rank == 0:
                    # torch.save(embedding_model.state_dict(), round_model_save_path_spk)
                    # torch.save(classifier.state_dict(), round_classifier_save_path_spk)
                    # with open(os.path.join(round_pseudo_label_dir, 'alabels_embeddings.pkl'), 'wb') as f:
                    #     pickle.dump(embeddings_dic, f)
                    if args.from_preds:
                        best_preds_dic_spk = {k: preds_dic[k] if k in audio_seg_ids_unreliable else audio_obs_init_results[k] for k in preds_dic}
                        best_uncertainty_dic_spk = copy.deepcopy(uncertainty_dic)
                        best_potential_list_dic_spk = copy.deepcopy(potential_list_dic)
                        with open(os.path.join(round_pseudo_label_dir, 'alabels_pred_dic.pkl'), 'wb') as f:
                            pickle.dump(best_preds_dic_spk, f)
                        with open(os.path.join(round_pseudo_label_dir, 'alabels_unreliable_dic.pkl'), 'wb') as f:
                            pickle.dump(best_uncertainty_dic_spk, f)
                        with open(os.path.join(round_pseudo_label_dir, 'alabels_potential_dic.pkl'), 'wb') as f:
                            pickle.dump(best_potential_list_dic_spk, f)

                if world_size > 1:
                    dist.barrier()
            else:
                patience_counter_epoch_spk += 1
                logger.info(f"Round {round}, Speaker Fine-tune Epoch {ft_epoch}: No improvement in validation accuracy. Patience(epoch): {patience_counter_epoch_spk}/{args.early_stop_patience_epoch}")
                if ft_epoch>2:
                    if (patience_counter_epoch_spk >= args.early_stop_patience_epoch):
                        logger.info(f"Early stopping at epoch {ft_epoch} due to no improvement in validation accuracy for {args.early_stop_patience_epoch} epochs.")
                        break
                    if (crt_acc_valid_e_pseudo_spk - prev_acc_valid_e_pseudo_spk) < -0.05:
                        logger.info(f"Early stopping at epoch {ft_epoch} due to significant drop in pseudo-label validation accuracy: {prev_acc_valid_e_pseudo_spk:.4f} -> {crt_acc_valid_e_pseudo_spk:.4f}.")
                        break
            prev_acc_valid_e_pseudo_spk = copy.deepcopy(crt_acc_valid_e_pseudo_spk)
        optimizer_spk.zero_grad()
        logger.info(f"Round {round}: Best speaker fine-tuned epoch={best_epoch_spk}, acc(valid_pseudo)={best_acc_valid_e_pseudo_spk:.4f}")

        # ============================
        # Face Model Fine-tuning
        # ============================
        if args.cluster_type == 'audio_vision' and face_train_loader is not None:
            # Unfreeze face embedding model's output_layer
            unfrozen_model_face_modules(face_embedding_model, print_mod_flag=(round == 0))
            # Optimizer for face fine-tuning (both embedding and classifier)
            optimizer_face = torch.optim.Adam(
                list(face_embedding_model.parameters()) + list(face_classifier.parameters()),
                lr=args.finetune_lr
            )
            logger.info(f"Fine-tuning face embedding model for {args.max_finetune_epochs} epochs...")

            # Fine-tuning
            round_model_save_path_face = os.path.join(round_dir, 'finetuned_model_face.pth')
            round_classifier_save_path_face = os.path.join(round_dir, 'finetuned_classifier_face.pth')
            best_acc_valid_e_pseudo_face, best_epoch_face, patience_counter_epoch_face = 0.0, 0, 0
            prev_acc_valid_e_pseudo_face = 0.0
            prev_acc_e_face = 0.0
            if args.from_preds:
                best_preds_dic_face, best_uncertainty_dic_face, best_potential_list_dic_face = {}, {}, {}
            
            for ft_epoch_face in range(args.max_finetune_epochs):
                torch.manual_seed(args.seed + ft_epoch_face)
                if args.use_gpu:
                    torch.cuda.manual_seed_all(args.seed + ft_epoch_face)
                train_stats_face = train_one_epoch(face_train_loader,  face_embedding_model, face_classifier, optimizer_face, 
                                              ft_epoch_face, logger, device, use_hidfeat_flag=False)
                logger.info(f"Round {round}, Face Fine-tune Epoch {ft_epoch_face}: loss={train_stats_face['loss']:.4f}, acc(pseudo)={train_stats_face['acc']:.2f}%")
                
                # Evaluate on annotated data and save predictions
                # Use unified inference function
                preds_dic_face, embeddings_dic_face, uncertainty_dic_face, potential_list_dic_face = inference_with_classifier(
                    model=face_embedding_model,
                    classifier=face_classifier,
                    dataset=face_train_dataset,
                    device=device,
                    compute_uncertainty=args.from_preds,
                    batch_process_flag=True,
                    batch_size=args.finetune_batch_size * 2,
                    use_hidfeat=False,
                    unfrozen_modules_num=None,
                    potential_set=train_dataset.unique_labels,
                )
                
                # Save predictions and compute accuracy on annotated data
                epoch_dir_face = os.path.join(round_dir, f'ft_epoch_face_{ft_epoch_face}')
                os.makedirs(epoch_dir_face, exist_ok=True)
                with open(os.path.join(epoch_dir_face, 'pseudo_labels_faces_mid_frame_pred.json'), 'w') as f:
                    json.dump(preds_dic_face, f, indent=2)
                crt_acc_valid_e_pseudo_face = compute_acc_from_dic(preds_dic_face, pseudo_valid_label_dic_face)
                crt_acc_e_face = compute_acc_from_anno(epoch_dir_face, args.face_anno_file, mode='all', modal='face')
                logger.info(f"Round {round}, Face Fine-tune Epoch {ft_epoch_face}: acc(valid_pseudo): {crt_acc_valid_e_pseudo_face:.4f}, acc: {crt_acc_e_face:.4f}")
                
                if crt_acc_valid_e_pseudo_face > best_acc_valid_e_pseudo_face:
                    best_acc_valid_e_pseudo_face, best_epoch_face = crt_acc_valid_e_pseudo_face, ft_epoch_face
                    patience_counter_epoch_face = 0
                    # Save best face model
                    if rank == 0:
                        # torch.save(face_embedding_model.state_dict(), round_model_save_path_face)
                        # torch.save(face_classifier.state_dict(), round_classifier_save_path_face)
                        # with open(os.path.join(round_pseudo_label_dir, 'vlabels_mf_embeddings.pkl'), 'wb') as f:
                        #     pickle.dump(embeddings_dic_face, f)
                        if args.from_preds:
                            best_preds_dic_face = copy.deepcopy(preds_dic_face)
                            best_uncertainty_dic_face = copy.deepcopy(uncertainty_dic_face)
                            best_potential_list_dic_face = copy.deepcopy(potential_list_dic_face)
                            with open(os.path.join(round_pseudo_label_dir, 'vlabels_mf_pred_dic.pkl'), 'wb') as f:
                                pickle.dump(best_preds_dic_face, f)
                            with open(os.path.join(round_pseudo_label_dir, 'vlabels_mf_unreliable_dic.pkl'), 'wb') as f:
                                pickle.dump(best_uncertainty_dic_face, f)
                            with open(os.path.join(round_pseudo_label_dir, 'vlabels_mf_potential_dic.pkl'), 'wb') as f:
                                pickle.dump(best_potential_list_dic_face, f)
                    if world_size > 1:
                        dist.barrier()
                else:
                    patience_counter_epoch_face += 1
                    logger.info(f"Round {round}, Face Fine-tune Epoch {ft_epoch_face}: No improvement. Patience(epoch): {patience_counter_epoch_face}/{args.early_stop_patience_epoch}")
                    if ft_epoch_face > 2:
                        if patience_counter_epoch_face >= args.early_stop_patience_epoch:
                            logger.info(f"Early stopping at face epoch {ft_epoch_face}")
                            break
                        if (crt_acc_valid_e_pseudo_face - prev_acc_valid_e_pseudo_face) < -0.05:
                            logger.info(f"Early stopping at face epoch {ft_epoch_face} due to significant drop: {prev_acc_valid_e_pseudo_face:.4f} -> {crt_acc_valid_e_pseudo_face:.4f}")
                            break
                prev_acc_valid_e_pseudo_face = copy.deepcopy(crt_acc_valid_e_pseudo_face)
            
            optimizer_face.zero_grad()
            logger.info(f"Round {round}: Best face fine-tuned epoch={best_epoch_face}, acc(valid_pseudo)={best_acc_valid_e_pseudo_face:.4f}")
        
        # ============================
        # Part 2: Extract embeddings(if use clustering to get pseudo labels)
        # ============================
        if not args.from_preds:
            logger.info(f"Round {round}: Extracting speaker embeddings")
            embs_dir = os.path.join(round_dir, 'embeddings')
            os.makedirs(embs_dir, exist_ok=True)
            if rank == 0:
                extract_embeddings_with_model(args.speaker_model_id, round_model_save_path_spk, args.conf, args.subseg_json,
                                              embs_dir, args.use_gpu, args.gpu)
            if world_size > 1:
                dist.barrier()
        else:
            shutil.copy(os.path.join(os.path.join(initial_dir, 'pseudo_label', 'useful_var_dic.pkl')), os.path.join(round_pseudo_label_dir, 'useful_var_dic.pkl'))
            embs_dir = ""

        # ============================
        # Part 3: Generate pseudo labels and evaluate
        # ============================
        # get HMM model from previous round
            hmm_model_path = os.path.join(pseudo_label_dir, 'hmm_params.pkl')
        if not os.path.exists(hmm_model_path):
            hmm_model_path = None
        else:
            logger.info(f"Using HMM model from previous round: {hmm_model_path}")
        # update pseudo_label_dir for current round to save results
        pseudo_label_dir = copy.deepcopy(round_pseudo_label_dir)
        
        if rank == 0:
            crt_acc_r_spk, crt_acc_r_face = run_clustering_and_evaluation(
                args.conf, args.cluster_type, args.wavs, embs_dir, args.visual_embs_dir, 
                pseudo_label_dir, args.use_hmm_smoothing, args.fix_mf,
                args.hmm_visual_info_type,  args.unreliable_pp,
                args.speaker_anno_file, args.face_anno_file, hmm_model_path, args.from_preds, mode='all')

            logger.info(f"Round {round}: speaker_acc={crt_acc_r_spk:.4f}")
            acc_history_spk.append({'round': round, 'acc': crt_acc_r_spk})
            with open(os.path.join(exp_dir, 'accuracy_history_speaker.json'), 'w') as f:
                json.dump(acc_history_spk, f, indent=2)
            if crt_acc_r_face is not None:
                logger.info(f"Round {round}: face_acc={crt_acc_r_face:.4f}")
                acc_history_face.append({'round': round, 'acc': crt_acc_r_face})
                with open(os.path.join(exp_dir, 'accuracy_history_face.json'), 'w') as f:
                    json.dump(acc_history_face, f, indent=2)
            
            # # Clean up previous checkpoint to save space
            # if round > 0:
            #     prev_round_dir = os.path.join(exp_dir, f'round{round-1}')
            #     prev_model = os.path.join(prev_round_dir, 'finetuned_model.pth')
            #     if os.path.exists(prev_model):
            #         os.remove(prev_model)
            #         logger.info(f"Removed previous checkpoint: {prev_model}")
        
        if world_size > 1:
            dist.barrier()
    
    # Final summary
    if rank == 0:
        logger.info("="*20)
        logger.info("Self-supervised fine-tuning completed!")
        logger.info(f"Initial speaker: acc={initial_acc_spk:.4f}")
        logger.info(f"Final speaker at round {round}: acc={crt_acc_r_spk:.4f}")
        logger.info(f"Speaker improvement: {(crt_acc_r_spk - initial_acc_spk):.4f} ({(crt_acc_r_spk - initial_acc_spk) / initial_acc_spk * 100:.2f}%)")
        if initial_acc_face is not None and crt_acc_r_face is not None:
            logger.info(f"Initial face: acc={initial_acc_face:.4f}")
            logger.info(f"Final face at round {round}: acc={crt_acc_r_face:.4f}")
            logger.info(f"Face improvement: {(crt_acc_r_face - initial_acc_face):.4f} ({(crt_acc_r_face - initial_acc_face) / initial_acc_face * 100:.2f}%)")
        logger.info("="*20)


if __name__ == "__main__":
    main()
