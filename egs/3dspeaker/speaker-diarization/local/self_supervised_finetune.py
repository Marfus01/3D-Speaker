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
import json, argparse, random, pickle
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
from cluster_and_postprocess import save_cluster_results_audio


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
        
        # Load pseudo-labels
        with open(pseudo_labels_json, 'r') as f:
            self.pseudo_labels = json.load(f)
        
        # Assert that the keys in subseg_info and pseudo_labels match
        assert set(self.subseg_info.keys()) == set(self.pseudo_labels.keys()), \
            "Keys in subseg_info and pseudo_labels do not match!"
        
        # 不再过滤-1标签，将所有样本都用于训练
        self.sample_ids = list(self.subseg_info.keys())
        
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

        # Create label encoder (map cluster IDs to continuous class indices)
        unique_labels = sorted(list(set([self.pseudo_labels[sid] for sid in self.sample_ids])))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.num_classes = len(unique_labels)
        self.class_weights = self.get_class_weights()
        self.label_weights = {self.idx2label[idx]: self.class_weights[idx].item() for idx in range(self.num_classes)}
        
        print(f"[INFO] Created dataset with {len(self.sample_ids)} samples and {self.num_classes} classes")
        print(f"[INFO] Label weights: {self.label_weights}")
    
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
        class_weights = 1.0 / (class_counts)**0.5
        class_weights = class_weights / np.sum(class_weights) * self.num_classes  # 归一化
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

def unfrozen_model_modules(model, unfrozen_modules_num=1, print_mod_flag=False):
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


def train_one_epoch(train_loader, model, classifier, optimizer, epoch, logger, device, use_hidfeat_flag=False, unfrozen_modules_num=1):
    """
    Train for one epoch without gradient accumulation.
    
    Args:
        use_hidfeat_flag: If True, input is hidden features and use forward_from
        unfrozen_modules_num: Number of unfrozen modules from top. always be 1 when use_hidfeat_flag is True, since the output of former modules have different dimension.(when use unfrozen_model_layers, unfrozen_layers_num needn't to be known here)
    """
    # criterion = nn.CrossEntropyLoss()
    class_weights = torch.tensor(train_loader.dataset.class_weights, dtype=torch.float32).to(device)
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

def cmd_compute_acc_spk(result_dir, speaker_anno_file, mode):
    """
    Call compute_acc_spk.py to compute speaker recognition accuracy.
    This is a simplified version that calls compute_acc_spk.py
    
    Returns:
        accuracy: Overall speaker recognition accuracy
    """
    # Run compute_acc_spk.py
    cmd = [
        sys.executable,
        'local/compute_acc_spk.py',
        '--result_dir', result_dir,
        '--ref_xlsx', speaker_anno_file,
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

def compute_speaker_accuracy(result_dir, speaker_anno_file, mode='valid'):
    """
    Compute speaker recognition accuracy from pseudo labels.
    This is a simplified version that calls compute_acc_spk.py
    
    Returns:
        accuracy: Overall speaker recognition accuracy
    """
    # check whether result_dir contains pseudo label json files
    pseudo_label_files = [f for f in os.listdir(result_dir) if f.endswith('.json') and 'pseudo_labels_audio' in f]
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
            cmd_compute_acc_spk(pseudo_labels_all_dir, speaker_anno_file, mode='valid')
            pseudo_label_files_acc_val = [get_acc(os.path.join(pseudo_labels_all_dir, f.replace('.json', '_accuracy(valid).txt'))) for f in pseudo_label_files]
            best_idx = int(np.argmax(np.array(pseudo_label_files_acc_val)))
            print(f"[INFO] Selected pseudo-label file {pseudo_label_files[best_idx]} with highest valid accuracy {pseudo_label_files_acc_val[best_idx]:.4f}")
            cmd_compute_acc_spk(pseudo_labels_all_dir, speaker_anno_file, mode='test') # used for logging purpose
        
        # Move the pseudo label file in pseudo_labels_all_dir with highest valid acc to result_dir
        src_path = os.path.join(pseudo_labels_all_dir, pseudo_label_files[best_idx])
        dst_path = os.path.join(result_dir, pseudo_label_files[best_idx])
        shutil.move(src_path, dst_path)
    
    # Run compute_acc_spk.py
    cmd_compute_acc_spk(result_dir, speaker_anno_file, mode)

    # Parse accuracy from the output file
    # Find the accuracy file that contains "corrected_all_by_HMM"
    acc_files = [f for f in os.listdir(result_dir) if f.endswith(f'_accuracy({mode}).txt') and 'pseudo_labels_audio' in f]
    assert len(acc_files) > 0, f"No accuracy file found in {result_dir}"
    assert len(acc_files) == 1, f"Multiple accuracy files found in {result_dir}: {acc_files}"
    acc = get_acc(os.path.join(result_dir, acc_files[0]))
    return acc


def run_clustering_and_evaluation(conf_file, cluster_type, wavs, audio_embs_dir, visual_embs_dir, result_dir, hmm_flag, fix_mf_flag, hmm_visual_info_type, unreliable_pp, speaker_anno_file, hmm_model_path=None, from_preds=False, mode='test'):
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
        hmm_model_path: Path to HMM model (optional)
        from_preds: Whether to use local predictions from classifier model instead of clustering (optional)
        mode: Mode for accuracy computation ('valid' or 'test')
    
    Returns:
        accuracy: Speaker recognition accuracy
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
    
    # Compute accuracy
    assert os.path.exists(speaker_anno_file), f"Speaker annotation file {speaker_anno_file} does not exist!"
    acc = compute_speaker_accuracy(result_dir, speaker_anno_file, mode)
    
    return acc


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
    ## Paths for best model and info
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    best_model_info_path = os.path.join(exp_dir, 'best_model_info.txt')
    best_classifier_path = os.path.join(exp_dir, 'best_classifier.pth')
    
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
        initial_acc_test = compute_speaker_accuracy(pseudo_label_dir, args.speaker_anno_file, mode='test')
        initial_acc_valid = compute_speaker_accuracy(pseudo_label_dir, args.speaker_anno_file, mode='valid')
    else:
        initial_acc_test = run_clustering_and_evaluation(
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
            hmm_model_path=None,
            from_preds=False,
            mode='test'
        )
        initial_acc_valid = compute_speaker_accuracy(pseudo_label_dir, args.speaker_anno_file, mode='valid')    # 写两次的目的是保证验证集acc文件也被复制到initial_dir中
        shutil.copytree(initial_dir, os.path.join(finetune_dir, 'initial'), dirs_exist_ok=True)
        logger.info(f"Saved initial clustering results to {initial_dir}")
    
    logger.info(f"Initial: acc(valid)={initial_acc_valid:.4f}, acc(test)={initial_acc_test:.4f}")
    # Accuracy history
    acc_history = [{'round': 'Initial', 'acc(valid)': initial_acc_valid, 'acc(test)': initial_acc_test}]
    
    # Load unreliable segment IDs and initial cluster results
    with open(os.path.join(pseudo_label_dir, 'useful_var_dic.pkl'), 'rb') as f:
        useful_var_dic = pickle.load(f)
    audio_seg_ids = useful_var_dic['audio_seg_ids']
    audio_cluster_unreliable_metrics = useful_var_dic['alabels_unreliable_metrics']
    idxs_unreliable = np.argsort(audio_cluster_unreliable_metrics)[:int(args.unreliable_pp / 100 * len(audio_seg_ids))]
    audio_seg_ids_unreliable = audio_seg_ids[idxs_unreliable]
    
    audio_cluster_result_files = [f for f in os.listdir(pseudo_label_dir) if f.endswith('.json') and 'cluster_results_audio_processed' in f]
    assert len(audio_cluster_result_files) == 1, f"No or multiple cluster_result_processed file found in {pseudo_label_dir}: {audio_cluster_result_files}"
    cluster_result_file = os.path.join(pseudo_label_dir, audio_cluster_result_files[0])
    with open(cluster_result_file, 'r') as f:
        initial_cluster_results = json.load(f)

    # Record best accuracy
    best_acc_valid_r = initial_acc_valid
    best_round = 0
    acc_test_at_best_valid = initial_acc_test
    patience_counter_round = 0
    
    # ============================
    # Iterative fine-tuning
    # ============================
    for round in range(args.max_rounds):
        logger.info("="*20)
        logger.info(f"Round {round}: Fine-tuning iteration")
        logger.info("="*20)
        
        round_dir = os.path.join(exp_dir, f'round{round}')
        os.makedirs(round_dir, exist_ok=True)
        
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
            # get objects of feature extractor and embedding model
            conf_model = update_conf(yaml_config_loader(args.conf), args.speaker_model_id, args.speaker_pretrained_model, rank)
            config_model = Config(conf_model)
            feature_extractor = build('feature_extractor', config_model)
            embedding_model = build('embedding_model', config_model)
            
            # get output embedding dimension of the embedding model
            embedding_dim = conf_model['embedding_model']['args']['embedding_size']
            logger.info(f"Embedding dimension: {embedding_dim}")

            # load pretrained model
            pretrained_state = torch.load(config_model.pretrained_model, map_location='cpu')
            embedding_model.load_state_dict(pretrained_state)
            embedding_model.to(device)
            # Check if the device is GPU
            if device.type == 'cuda':
                print(f"[INFO]: Using GPU: {torch.cuda.get_device_name(device)}")
            else:
                print("[INFO]: Using CPU")

            # Save config_model as JSON
            config_model_json_path = os.path.join(round_dir, 'config_model.json')
            with open(config_model_json_path, 'w') as f:
              json.dump(conf_model, f, indent=4)  # Assuming conf_model is serializable
            logger.info(f"Saved config_model to {config_model_json_path}")
            
            # Copy pretrained model to local round dir for record
            torch.save(embedding_model.state_dict(), best_model_path)
            # Write best model info to file
            with open(best_model_info_path, 'a') as f:
              f.write(f"Initial: acc(valid)={initial_acc_valid:.4f}, acc(test)={initial_acc_test:.4f}\n")

        # Create dataset and dataloader
        ## Define subseg_hidfeat_dic at round 0
        if round == 0:
            train_dataset = PseudoLabelDataset(args.subseg_json, pseudo_label_file, feature_extractor)
            crt_collate_fn = collate_fn_hidden if args.use_hidfeat else collate_fn
            subseg_hidfeat_dic = None
            if args.use_hidfeat:
                logger.info(f"Extracting hidden features for all samples,which are outputs of StatsPool layer...")
                logger.info(f"args.unfrozen_layers_num={args.unfrozen_layers_num} is not used when extracting hidden features.")
                subseg_hidfeat_dic = {}
                embedding_model.eval()
                with torch.no_grad():
                    for sid in train_dataset.sample_ids:
                        feat = train_dataset.subseg_feat_dic[sid].unsqueeze(0).to(device)  # [1, T, F]
                        hidden_feat = embedding_model.forward_until(feat, 1)
                        subseg_hidfeat_dic[sid] = hidden_feat.squeeze(0).detach().cpu()  # [hid_dim]
                logger.info(f"Hidden features extracted for {len(subseg_hidfeat_dic)} samples")
        
        ## Re-create dataset with new pseudo_label_file
        train_dataset = PseudoLabelDataset(args.subseg_json, pseudo_label_file, feature_extractor, args.use_hidfeat, subseg_hidfeat_dic)
        
        if len(train_dataset) == 0:
            logger.error("No valid training samples found!")
            break
        batch_sampler = LengthAwareBatchSampler(train_dataset, batch_size=args.finetune_batch_size, shuffle=True, use_hidfeat_flag=args.use_hidfeat)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, collate_fn=crt_collate_fn)

        # Freeze embedding model initially
        for param in embedding_model.parameters():
            param.requires_grad = False
        
        # Create classifier and warm up
        if round == 0 or not args.from_preds: # 当根据分类结果获取伪标签时，每个round伪标签都与之前类似
            # Create classifier
            classifier = MLPClassifier(input_dim=embedding_dim, num_classes=train_dataset.num_classes).to(device)
        
            # Warmup: train classifier only when round == 0
            ## Optimizer for warmup (only classifier).
            optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
            ## Warmup training
            logger.info(f"Warmup training classifier for {args.warmup_epochs_num} epochs...")
            for warmup_epoch in range(args.warmup_epochs_num):
                torch.manual_seed(args.seed + warmup_epoch)
                if args.use_gpu:
                    torch.cuda.manual_seed_all(args.seed + warmup_epoch)
                train_stats = train_one_epoch(train_loader,  embedding_model, classifier, optimizer, 
                                              warmup_epoch, logger, device, args.use_hidfeat)
                logger.info(f"Round {round}, Warmup Epoch {warmup_epoch}: loss={train_stats['loss']:.4f}, acc(pseudo)={train_stats['acc']:.2f}%")
            optimizer.zero_grad()

        # Unfreeze last few layers in embedding model
        if args.use_hidfeat:
            if round == 0:
                unfrozen_model_modules(embedding_model, unfrozen_modules_num=1, print_mod_flag=True)
            else:
                unfrozen_model_modules(embedding_model, unfrozen_modules_num=1)
        else: 
            if round == 0:
                unfrozen_model_layers(embedding_model, args.unfrozen_layers_num, print_mod_flag=True)
            else:
                unfrozen_model_layers(embedding_model, args.unfrozen_layers_num)

        # Optimizer for fine-tuning (both embedding and classifier)
        optimizer = torch.optim.Adam(
            list(embedding_model.parameters()) + list(classifier.parameters()),
            lr=args.finetune_lr
        )
        
        # Fine-tuning
        round_pseudo_label_dir = os.path.join(round_dir, 'pseudo_label')
        os.makedirs(round_pseudo_label_dir, exist_ok=True)
        round_model_save_path = os.path.join(round_dir, 'finetuned_model.pth')
        round_classifier_save_path = os.path.join(round_dir, 'finetuned_classifier.pth')
        best_acc_valid_e, best_epoch, patience_counter_epoch = 0.0, 0, 0
        if args.from_preds:
            best_preds_dic, best_uncertainty_dic, best_potential_list_dic = {}, {}, {}
        logger.info(f"Fine-tuning embedding model for {args.max_finetune_epochs} epochs...")
        for ft_epoch in range(args.max_finetune_epochs):
            torch.manual_seed(args.seed + ft_epoch)
            if args.use_gpu:
                torch.cuda.manual_seed_all(args.seed + ft_epoch)
            train_stats = train_one_epoch(train_loader,  embedding_model, classifier, optimizer, 
                                          ft_epoch, logger, device, args.use_hidfeat)
            logger.info(f"Round {round}, Fine-tune Epoch {ft_epoch}: loss={train_stats['loss']:.4f}, acc(pseudo)={train_stats['acc']:.2f}%")
            
            # === 每个epoch后，计算所有样本的分类标签、概率和不确定度 ===
            embedding_model.eval()
            classifier.eval()
            preds_dic = {}
            if args.from_preds:
                potential_list_dic, uncertainty_dic = {}, {}
            # else:
            #     embeddings_dic = {}
            if args.use_hidfeat:  # 顺序读取 hidden features 的 batch
                # Create a dataloader for inference (no shuffling, sequential order)
                inference_loader = DataLoader(train_dataset, batch_size=args.finetune_batch_size * 2, 
                                              sampler=SequentialSampler(train_dataset),
                                              num_workers=4, pin_memory=True, collate_fn=collate_fn_hidden)
                
                with torch.no_grad():
                    sample_idx = 0
                    for batch_feat, _ in inference_loader:
                        batch_feat = batch_feat.to(device)  # [B, ...]
                        batch_size = batch_feat.size(0)
                        # Forward pass
                        batch_emb = embedding_model.forward_from(batch_feat, 1)
                        batch_logits = classifier(batch_emb)
                        batch_probs = torch.softmax(batch_logits, dim=1)  # [B, num_classes]
                        batch_pred_labels = torch.argmax(batch_probs, dim=1)  # [B]
                        
                        # Map back to sample IDs and store results
                        for i in range(batch_size):
                            sid = train_dataset.sample_ids[sample_idx]
                            pred_label = int(train_dataset.idx2label[batch_pred_labels[i].item()])
                            preds_dic[sid] = pred_label if sid in audio_seg_ids_unreliable else initial_cluster_results[sid]
                            
                            if args.from_preds:
                                probs = batch_probs[i]
                                top2_probs, top2_indices = torch.topk(probs, 2)
                                uncertainty = (top2_probs[0] - top2_probs[1]).item() if sid in audio_seg_ids_unreliable else 2
                                uncertainty_dic[sid] = float(uncertainty)
                                potential_list = top2_indices.cpu().numpy().tolist()
                                potential_list = [int(train_dataset.idx2label[idx]) for idx in potential_list]
                                potential_list_dic[sid] = potential_list
                            # else:
                            #     emb = batch_emb[i]
                            #     embeddings_dic[sid] = emb.squeeze(0).cpu().numpy()
                            sample_idx += 1

            else: # 遍历所有subseg_id，取原始特征
                with torch.no_grad():
                    for sid in train_dataset.sample_ids:
                        # get probs of all classes
                        feat = train_dataset.subseg_feat_dic[sid]  # 原始特征，无截断
                        feat = feat.unsqueeze(0).to(device)  # [1, T, D]
                        emb = embedding_model(feat)  # [1, emb_dim]
                        logits = classifier(emb)
                        probs = torch.softmax(logits, dim=1)[0]  # [num_classes]
                        # get predicted label and save to dict
                        pred_label = torch.argmax(probs).item()
                        pred_label = int(train_dataset.idx2label[pred_label])
                        preds_dic[sid] = pred_label if sid in audio_seg_ids_unreliable else initial_cluster_results[sid]
                        # get potential labels and uncertainty and save to dict
                        if args.from_preds:
                            top2_probs, top2_indices = torch.topk(probs, 2)
                            uncertainty = (top2_probs[0] - top2_probs[1]).item() if sid in audio_seg_ids_unreliable else 2
                            uncertainty_dic[sid] = float(uncertainty)
                            potential_list = top2_indices.cpu().numpy().tolist()
                            potential_list = [int(train_dataset.idx2label[idx]) for idx in potential_list]
                            potential_list_dic[sid] = potential_list
                        # else:
                        #     embeddings_dic[sid] = emb.squeeze(0).cpu().numpy()


            # 计算在验证集上的准确率
            epoch_dir = os.path.join(round_dir, f'ft_epoch_{ft_epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            save_cluster_results_audio(np.array([preds_dic[k] for k in train_dataset.sample_ids]), np.array(train_dataset.sample_ids), os.path.join(epoch_dir, f'pseudo_labels_audio_pred.json'))
            crt_acc_valid_e = compute_speaker_accuracy(epoch_dir, args.speaker_anno_file, mode='valid')
            logger.info(f"Round {round}, Fine-tune Epoch {ft_epoch}: acc(valid): {crt_acc_valid_e:.4f}")
            if crt_acc_valid_e > best_acc_valid_e:  # epoch0 must be better
                best_acc_valid_e, best_epoch  = crt_acc_valid_e, ft_epoch
                patience_counter_epoch = 0
                # Save best model of this round, and preds, uncertainty, potential_list dicts
                if rank == 0:
                    torch.save(embedding_model.state_dict(), round_model_save_path)
                    torch.save(classifier.state_dict(), round_classifier_save_path)
                    if args.from_preds:
                        best_preds_dic = copy.deepcopy(preds_dic)
                        best_uncertainty_dic = copy.deepcopy(uncertainty_dic)
                        best_potential_list_dic = copy.deepcopy(potential_list_dic)
                        with open(os.path.join(round_pseudo_label_dir, 'alabels_pred_dic.pkl'), 'wb') as f:
                            pickle.dump(best_preds_dic, f)
                        with open(os.path.join(round_pseudo_label_dir, 'alabels_unreliable_dic.pkl'), 'wb') as f:
                            pickle.dump(best_uncertainty_dic, f)
                        with open(os.path.join(round_pseudo_label_dir, 'alabels_potential_dic.pkl'), 'wb') as f:
                            pickle.dump(best_potential_list_dic, f)
                    # else:
                    #     with open(os.path.join(round_pseudo_label_dir, 'embeddings.pkl'), 'wb') as f:
                    #         pickle.dump(embeddings_dic, f)
                if world_size > 1:
                    dist.barrier()
            else:
                patience_counter_epoch += 1
                logger.info(f"Round {round}, Fine-tune Epoch {ft_epoch}: No improvement in validation accuracy. Patience(epoch): {patience_counter_epoch}/{args.early_stop_patience_epoch}")
                if patience_counter_epoch >= args.early_stop_patience_epoch:
                    logger.info(f"Early stopping at epoch {ft_epoch} due to no improvement in validation accuracy for {args.early_stop_patience_epoch} epochs.")
                    break
        optimizer.zero_grad()
        logger.info(f"Round {round}: Best fine-tuned epoch={best_epoch}, acc(valid)={best_acc_valid_e:.4f}")
        
        
        # ============================
        # Part 2: Extract embeddings(if use clustering to get pseudo labels)
        # ============================
        if not args.from_preds:
            logger.info(f"Round {round}: Extracting embeddings")
            embs_dir = os.path.join(round_dir, 'embeddings')
            os.makedirs(embs_dir, exist_ok=True)
            if rank == 0:
                extract_embeddings_with_model(args.speaker_model_id, round_model_save_path, args.conf, args.subseg_json,
                                              embs_dir, args.use_gpu, args.gpu)
            if world_size > 1:
                dist.barrier()
        else:
            shutil.copy(os.path.join(os.path.join(initial_dir, 'pseudo_label', 'useful_var_dic.pkl')), os.path.join(round_dir, 'pseudo_label', 'useful_var_dic.pkl'))
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
        pseudo_label_dir = round_pseudo_label_dir
        
        if rank == 0:
            crt_acc_test_r = run_clustering_and_evaluation(args.conf, args.cluster_type, args.wavs, embs_dir, args.visual_embs_dir, 
                                                        pseudo_label_dir, args.use_hmm_smoothing, args.fix_mf,
                                                        args.hmm_visual_info_type,  args.unreliable_pp,
                                                        args.speaker_anno_file, hmm_model_path, args.from_preds, mode='test')
            crt_acc_valid_r = compute_speaker_accuracy(pseudo_label_dir, args.speaker_anno_file, mode='valid')
            logger.info(f"Round {round}: acc(valid)={crt_acc_valid_r:.4f}, acc(test)={crt_acc_test_r:.4f}")
            # Record accuracy
            acc_history.append({'round': round, 'acc(valid)': crt_acc_valid_r, 'acc(test)': crt_acc_test_r})
            
            # Check if best
            if crt_acc_valid_r > best_acc_valid_r:
                # Update best accuracy
                best_acc_valid_r, best_round, acc_test_at_best_valid = crt_acc_valid_r, round, crt_acc_test_r
                patience_counter_round = 0
                # Save best model
                shutil.copy(round_model_save_path, best_model_path)
                shutil.copy(round_classifier_save_path, best_classifier_path)
                # Write best model info to file
                with open(best_model_info_path, 'a') as f:
                    f.write(f"Round {round}: acc(valid)={crt_acc_valid_r:.4f}, acc(test)={crt_acc_test_r:.4f}\n") 
                logger.info(f"New best acc(valid) at round {best_round}: acc(valid)={crt_acc_valid_r:.4f}, acc(test)={crt_acc_test_r:.4f}") 
            else:
                patience_counter_round += 1
                logger.info(f"No improvement. Patience(round): {patience_counter_round}/{args.early_stop_patience_round}")
            
            # Save accuracy history
            with open(os.path.join(exp_dir, 'accuracy_history.json'), 'w') as f:
                json.dump(acc_history, f, indent=2)
            # Early stopping
            if patience_counter_round >= args.early_stop_patience_round:
                logger.info(f"Early stopping triggered at round {best_round}.")
                break
            
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
        logger.info(f"Initial: acc(valid)={initial_acc_valid:.4f}, acc(test)={initial_acc_test:.4f}")
        logger.info(f"Best acc(valid) at round {best_round}: acc(valid)={best_acc_valid_r:.4f}, acc(test)={acc_test_at_best_valid:.4f}")
        logger.info(f"Improvement for acc(valid): {(best_acc_valid_r - initial_acc_valid):.4f} ({(best_acc_valid_r - initial_acc_valid) / initial_acc_valid * 100:.2f}%)")
        logger.info(f"Improvement for acc(test): {(acc_test_at_best_valid - initial_acc_test):.4f} ({(acc_test_at_best_valid - initial_acc_test) / initial_acc_test * 100:.2f}%)")
        logger.info("="*20)


if __name__ == "__main__":
    main()
