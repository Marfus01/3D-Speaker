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

import os, sys, time, shutil, subprocess
import json, argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

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
parser.add_argument('--max_rounds', default=10, type=int, help='Maximum fine-tuning epochs')
parser.add_argument('--early_stop_patience', default=5, type=int, help='Early stopping patience')
parser.add_argument('--finetune_lr', default=0.001, type=float, help='Fine-tuning learning rate')
parser.add_argument('--finetune_batch_size', default=64, type=int, help='Fine-tuning batch size')
parser.add_argument('--warmup_epochs_num', default=2, type=int, help='Warmup epochs for classifier head')
parser.add_argument('--finetune_epochs_num', default=5, type=int, help='Number of epochs for each fine-tuning stage')
parser.add_argument('--unfrozen_layers_num', default=2, type=int, help='Number of layers to unfreeze (from top) during fine-tuning')

# Distributed training
parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')
parser.add_argument('--seed', default=1234, type=int, help='Random seed')


class PseudoLabelDataset(Dataset):
    """
    Dataset for fine-tuning with pseudo-labels from pseudo labels.
    """
    def __init__(self, subseg_json, pseudo_labels_json, feature_extractor):
        """
        Args:
            subseg_json: Path to sub-segments json file
            pseudo_labels_json: Path to pseudo labels json file
            feature_extractor: Feature extraction object
        """
        self.feature_extractor = feature_extractor
        
        # Load sub-segments info
        with open(subseg_json, 'r') as f:
            self.subseg_info = json.load(f)
        
        # Load pseudo-labels
        with open(pseudo_labels_json, 'r') as f:
            self.pseudo_labels = json.load(f)
        
        # Assert that the keys in subseg_info and pseudo_labels match
        assert set(self.subseg_info.keys()) == set(self.pseudo_labels.keys()), \
            "Keys in subseg_info and pseudo_labels do not match!"
        
        # Filter valid samples (those with pseudo-labels and not labeled as -1)
        self.valid_samples = []
        for subseg_id, _ in self.subseg_info.items():
            if self.pseudo_labels[subseg_id] >= 0:  # Exclude noise class (-1)
                self.valid_samples.append(subseg_id)
        
        # Get all wav data to speed up loading
        obj_fs = self.feature_extractor.sample_rate
        self.wav_paths = list(set([self.subseg_info[sid]['file'] for sid in self.valid_samples]))
        self.wav_dat_dic = {wav_path: load_audio(wav_path, obj_fs=obj_fs) for wav_path in self.wav_paths}
        self.subseg_wav_dic = {sid: self.wav_dat_dic[self.subseg_info[sid]['file']][0, int(self.subseg_info[sid]['start']*obj_fs):int(self.subseg_info[sid]['stop']*obj_fs)].unsqueeze(0) for sid in self.valid_samples}    # each elements is (1, num_samples_i))
        del self.wav_paths, self.wav_dat_dic
        
        # Create label encoder (map cluster IDs to continuous class indices)
        unique_labels = sorted(list(set([self.pseudo_labels[sid] for sid in self.valid_samples])))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        
        print(f"[INFO] Created dataset with {len(self.valid_samples)} samples and {self.num_classes} classes")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, index):
        subseg_id = self.valid_samples[index]
        # Load audio and extract mel feature
        waveform = self.subseg_wav_dic[subseg_id]   # don't need to add batch dim here
        feat = torch.vmap(self.feature_extractor)(waveform.unsqueeze(0)).squeeze(0) # convert segment to mel feature, [num_frames, n_mels]
        
        # Get pseudo-label
        cluster_label = self.pseudo_labels[subseg_id]
        class_idx = self.label_to_idx[cluster_label]
        
        return feat, class_idx


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


def unfrozen_model_layers(model, unfrozen_layers_num=0):
    """
    Unfreeze the top layers of the model.
    
    Args:
        model: The embedding model
        unfrozen_layers_num: Number of layers to unfreeze from the top
    """
    # Get all modules
    all_modules = list(model.named_modules())
    print(f"[INFO] Total layers in model: {len(all_modules)}")
    print(f"[INFO] All layers:")
    for idx, (name, module) in enumerate(all_modules):
        print(f"  [{idx}] {name}")
        print(f"    {module}")
    
    # Unfreeze top layers
    if unfrozen_layers_num > 0:
        for idx, (name, module) in enumerate(all_modules):
            if idx >= len(all_modules) - unfrozen_layers_num:
                for param in module.parameters():
                    param.requires_grad = True
                print(f"[INFO] Unfroze layer [{idx}] {name}")


def train_one_epoch(train_loader, bs, model, classifier, criterion, optimizer, epoch, logger, device):
    """
    Train for one epoch with gradient accumulation.
    """
    train_stats = AverageMeters()
    train_stats.add('Time', ':6.3f')
    train_stats.add('Loss', ':.4e')
    train_stats.add('Acc@1', ':6.2f')
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
        feat = feat.to(device)
        label = label.to(device)
        
        # Forward
        embedding = model(feat)
        output = classifier(embedding)
        loss = criterion(output, label) / bs  # Scale loss by accumulation steps
        acc1 = accuracy(output, label)
        # Backward
        loss.backward()
        
        # Update gradients every `bs` steps
        if (i + 1) % bs == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Record
        train_stats.update('Loss', loss.item() * bs, feat.size(0))  # Scale back loss for logging
        train_stats.update('Acc pseudo', acc1.item(), feat.size(0))
        train_stats.update('Lr', optimizer.param_groups[0]["lr"])
        train_stats.update('Time', time.time() - end)
        
        if i % 50 == 0:
            logger.info(progress.display(i))
        
        end = time.time()
    
    return {
        'loss': train_stats.avg('Loss'),
        'acc': train_stats.avg('Acc@1'),
        'lr': train_stats.val('Lr')
    }


def extract_embeddings_with_model(speaker_model_id, speaker_model_path, conf_file, subseg_json, audio_embs_out_dir, gpu, use_gpu):
    """
    通过命令行调用 extract_diar_embeddings.py，提取所有语音的 embedding 并保存到指定目录。

    Args:
        speaker_model_id: 说话人模型ID（如 iic/speech_campplus_sv_zh-cn_3dspeaker_16k）
        speaker_model_path: 微调后的说话人模型路径
        subseg_json: 子片段信息json
        audio_embs_out_dir: 输出embedding目录
        gpu: GPU id 列表（如 [0] 或 [0,1]）
        use_gpu: 是否使用GPU（bool）
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
        if gpu:
            if isinstance(gpu, (list, tuple)):
                gpu_str = ' '.join(str(g) for g in gpu)
            else:
                gpu_str = str(gpu)
            cmd.extend(['--gpu'] + gpu_str.split())

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


def compute_speaker_accuracy(result_dir, speaker_anno_file):
    """
    Compute speaker recognition accuracy from pseudo labels.
    This is a simplified version that calls compute_acc_spk.py
    
    Returns:
        accuracy: Overall speaker recognition accuracy
    """
    
    # Run compute_acc_spk.py
    cmd = [
        sys.executable,
        'local/compute_acc_spk.py',
        '--result_dir', result_dir,
        '--ref_xlsx', speaker_anno_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Error computing accuracy: {e}")
        return 0.0
    
    # Parse accuracy from the output file
    # Find the accuracy file that contains "corrected_all_by_HMM"
    acc_files = [f for f in os.listdir(result_dir) if f.endswith('_accuracy.txt') and 'pseudo_labels_audio' in f]
    assert len(acc_files) > 0, f"No accuracy file found in {result_dir}"
    assert len(acc_files) == 1, f"Multiple accuracy files found in {result_dir}: {acc_files}"
    
    # Read the most recent accuracy file
    acc_file = os.path.join(result_dir, acc_files[0])
    try:
        with open(acc_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('overall_accuracy'):
                    acc = float(line.split(':')[1].strip())
                    return acc
    except Exception as e:
        print(f"[WARNING] Error parsing accuracy file: {e}")
        return 0.0
    
    return 0.0


def run_clustering_and_evaluation(conf_file, cluster_type, wavs, audio_embs_dir, visual_embs_dir, result_dir, hmm_flag, fix_mf_flag, hmm_visual_info_type, unreliable_pp, speaker_anno_file):
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
        return 0.0
    
    # Compute accuracy
    assert os.path.exists(speaker_anno_file), f"Speaker annotation file {speaker_anno_file} does not exist!"
    acc = compute_speaker_accuracy(result_dir, speaker_anno_file)
    
    return acc


def main():
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
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
    os.makedirs(args.result_dir, exist_ok=True)
    finetune_dir = os.path.join(args.result_dir, 'self_supervised')
    os.makedirs(finetune_dir, exist_ok=True)
    # Save args to a dictionary and write to a JSON file
    args_json_path = os.path.join(finetune_dir, 'args.json')
    with open(args_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Saved arguments to {args_json_path}")
    
    # Setup logger
    logger = get_logger(os.path.join(finetune_dir, 'self_supervised_train.log'))
    logger.info(f"Starting self-supervised fine-tuning pipeline")
    logger.info(f"Device: {device}")
    
    # ============================
    # Round 0 Part 0: Initial clustering
    # ============================
    logger.info("="*20)
    logger.info("Round 0 Part 0: Initial clustering with HMM correction")
    logger.info("="*20)
    
    round0_part0_dir = os.path.join(finetune_dir, 'round0_part0')
    pseudo_label_dir = os.path.join(round0_part0_dir, 'pseudo_label')
    os.makedirs(pseudo_label_dir, exist_ok=True)
    
    initial_acc = run_clustering_and_evaluation(
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
        args.speaker_anno_file
    )
    
    logger.info(f"Round 0 Part 0 - Initial accuracy: {initial_acc:.4f}")
    
    # Record best accuracy
    best_acc = initial_acc
    best_round = 0
    patience_counter = 0
    
    # Accuracy history
    acc_history = [{'round': 0, 'part': 'part0', 'acc': initial_acc}]
    
    # ============================
    # Iterative fine-tuning
    # ============================
    for round in range(args.max_rounds):
        logger.info("="*20)
        logger.info(f"Round {round}: Fine-tuning iteration")
        logger.info("="*20)
        
        round_dir = os.path.join(finetune_dir, f'round{round}')
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
            
            # load pretrained model
            pretrained_state = torch.load(conf_model.pretrained_model, map_location='cpu')
            embedding_model.load_state_dict(pretrained_state)
            embedding_model.to(device)
            # Check if the device is GPU
            if device.type == 'cuda':
                print(f"[INFO]: Using GPU: {torch.cuda.get_device_name(device)}")
            else:
                print("[INFO]: Using CPU")
        
        # Create dataset and dataloader
        train_dataset = PseudoLabelDataset(args.subseg_json, pseudo_label_file, feature_extractor)
        if len(train_dataset) == 0:
            logger.error("No valid training samples found!")
            break
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if world_size > 1 else None
        train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True )
        
        # Create classifier with the dynamically obtained embedding size
        sample_feat, _ = next(iter(train_loader))
        with torch.no_grad():
            sample_embedding = embedding_model(sample_feat.to(device))
        embedding_dim = sample_embedding.shape[-1]
        logger.info(f"Embedding dimension: {embedding_dim}")
        classifier = MLPClassifier(input_dim=embedding_dim, num_classes=train_dataset.num_classes).to(device)
        
        # Warmup: train classifier only
        ## Freeze embedding model initially (warmup classifier)
        for param in embedding_model.parameters():
            param.requires_grad = False
        ## Optimizer for warmup (only classifier).
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        ## Warmup training
        logger.info(f"Warmup training classifier for {args.warmup_epochs_num} epochs...")
        for warmup_epoch in range(args.warmup_epochs_num):
            if train_sampler:
                train_sampler.set_epoch(warmup_epoch)
            train_stats = train_one_epoch(
                train_loader, args.finetune_batch_size, embedding_model, classifier, criterion,
                optimizer, warmup_epoch, logger, device
            )
            logger.info(f"Warmup Epoch {warmup_epoch}: Loss={train_stats['loss']:.4f}, Acc={train_stats['acc']:.2f}%")
        optimizer.zero_grad()

        # Unfreeze last few layers
        unfrozen_model_layers(embedding_model, args.unfrozen_layers_num)
        
        # Optimizer for fine-tuning (both embedding and classifier)
        optimizer = torch.optim.Adam(
            list(embedding_model.parameters()) + list(classifier.parameters()),
            lr=args.finetune_lr
        )
        
        # Fine-tuning
        logger.info(f"Fine-tuning embedding model for {args.finetune_epochs_num} epochs...")
        for ft_epoch in range(args.finetune_epochs_num):
            if train_sampler:
                train_sampler.set_epoch(ft_epoch + args.warmup_epochs_num)
            train_stats = train_one_epoch(
                train_loader, args.finetune_batch_size, embedding_model, classifier, criterion,
                optimizer, ft_epoch, logger, device
            )
            logger.info(f"Fine-tune Epoch {ft_epoch}: Loss={train_stats['loss']:.4f}, Acc={train_stats['acc']:.2f}%")
        optimizer.zero_grad()
        
        # Save fine-tuned model
        if rank == 0:
            model_save_path = os.path.join(round_dir, 'finetuned_model.pth')
            torch.save(embedding_model.state_dict(), model_save_path)
            logger.info(f"Saved fine-tuned model to {model_save_path}")
        
        if world_size > 1:
            dist.barrier()
        
        # ============================
        # Part 3: Extract embeddings, cluster, evaluate
        # ============================
        logger.info(f"Round {round} Part 3: Extracting embeddings and evaluating")
        
        # Extract embeddings
        embs_dir = os.path.join(round_dir, 'embeddings')
        os.makedirs(embs_dir, exist_ok=True)
        
        if rank == 0:
            extract_embeddings_with_model(args.speaker_model_id, model_save_path, args.conf, args.subseg_json,
                                          embs_dir, args.use_gpu, args.gpu)
        
        if world_size > 1:
            dist.barrier()
        
        # Run clustering and evaluation
        pseudo_label_dir = os.path.join(round_dir, 'pseudo_label')
        os.makedirs(pseudo_label_dir, exist_ok=True)
        
        if rank == 0:
            current_acc = run_clustering_and_evaluation(args.conf, args.cluster_type, args.wavs,embs_dir, args.visual_embs_dir, 
                                                        pseudo_label_dir, args.use_hmm_smoothing, args.fix_mf,
                                                        args.hmm_visual_info_type,  args.unreliable_pp,
                                                        args.speaker_anno_file)

            logger.info(f"Round {round} Part 3 - Accuracy: {current_acc:.4f}")
            acc_history.append({'round': round, 'part': 'part3', 'acc': current_acc})
            
            # Check if best
            if current_acc > best_acc:
                best_acc = current_acc
                best_round = round
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(finetune_dir, 'best_model.pth')
                shutil.copy(os.path.join(round_dir, 'finetuned_model.pth'), best_model_path)
                logger.info(f"New best accuracy: {best_acc:.4f} at round {best_round}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}")
            
            # Save accuracy history
            with open(os.path.join(finetune_dir, 'accuracy_history.json'), 'w') as f:
                json.dump(acc_history, f, indent=2)
            
            # Early stopping
            if patience_counter >= args.early_stop_patience:
                logger.info(f"Early stopping triggered. Best accuracy: {best_acc:.4f} at round {best_round}")
                break
            
            # # Clean up previous checkpoint to save space
            # if round > 0:
            #     prev_round_dir = os.path.join(finetune_dir, f'round{round-1}')
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
        logger.info(f"Initial accuracy: {initial_acc:.4f}")
        logger.info(f"Best accuracy: {best_acc:.4f} at round {best_round}")
        logger.info(f"Improvement: {(best_acc - initial_acc):.4f} ({(best_acc - initial_acc) / initial_acc * 100:.2f}%)")
        logger.info("="*20)


if __name__ == "__main__":
    main()
