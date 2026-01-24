# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
Contrastive learning pipeline for speaker embedding model.
This script implements contrastive learning fine-tuning on the current dataset
before extracting embeddings for speaker diarization.

Pipeline:
- Load pretrained speaker model
- Prepare contrastive learning dataset with augmentation
- Train with contrastive loss
- Evaluate EER and minDCF after each epoch
- Save best model based on EER
"""

import os, sys, time, glob, random, argparse, json
import soundfile
import numpy as np
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from pytorch_revgrad import RevGrad
from utils import read_testEER, evaluate_EER

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from speakerlab.utils.builder import build
from speakerlab.utils.fileio import load_audio
from speakerlab.utils.config import yaml_config_loader, Config
from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter
from extract_diar_embeddings import update_conf


parser = argparse.ArgumentParser(description='Contrastive learning for speaker diarization')
parser.add_argument('--conf', required=True, type=str, help='Config file')
parser.add_argument('--subseg_json', required=True, type=str, help='Sub-segments json file')
parser.add_argument('--musan_path', required=True, type=str, help='Path to MUSAN noise dataset')
parser.add_argument('--rir_filepath', required=True, type=str, help='Path to RIR noise file')
parser.add_argument('--testEER_file', required=True, type=str, help='Test EER file with sample pairs')
parser.add_argument('--result_dir', required=True, type=str, help='Result directory')

parser.add_argument('--speaker_model_id', default=None, help='Speaker model id in modelscope')
parser.add_argument('--speaker_pretrained_model', default=None, type=str, help='Path of local pretrained model')

# Contrastive learning parameters
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
parser.add_argument('--max_dur', default=2.0, type=float, help='Maximum duration per segment in seconds')
parser.add_argument('--max_epochs', default=100, type=int, help='Maximum training epochs')
parser.add_argument('--test_interval', default=1, type=int, help='Evaluation interval (epochs)')
parser.add_argument('--early_stop_patience', default=10, type=int, help='Early stopping patience')

# Distributed training
parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')
parser.add_argument('--seed', default=1234, type=int, help='Random seed')


class ContrastiveTrainDataset(Dataset):
    """
    Dataset for contrastive learning with augmentation.
    Loads two segments from each utterance and applies different augmentations.
    Returns augmented mel features(3, n_frames, n_mels) for each utterance.
    """
    def __init__(self, subseg_json, max_dur, musan_path, rir_filepath, feature_extractor):
        self.subseg_json = subseg_json
        self.max_dur = max_dur
        self.feature_extractor = feature_extractor
        self.obj_fs = self.feature_extractor.sample_rate

        # Load sub-segments info
        with open(subseg_json, 'r') as f:
            self.subseg_info = json.load(f)        
        self.sample_ids = list(self.subseg_info.keys()) # List of sample ids like "E01-1"
        self.wav_paths = list(set([self.subseg_info[sid]['file'] for sid in self.sample_ids]))
        self.wav_dat_dic = {wav_path: load_audio(wav_path, self.obj_fs) for wav_path in self.wav_paths}
        
        # Noise augmentation setup
        ## Load RIR files
        self.rir_files = np.load(rir_filepath)# rir noise data. np.arr of shape (1000, 11200)
        ## Load noise files from MUSAN
        ### Basic Attributes
        self.noisetypes = ['noise', 'speech', 'music'] # 3 Type of noise in MUSAN
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]} # The range of SNR
        ## Build noise file list based om all files in MUSAN
        self.noiselist = {} # key: noise type, value: noise filepath list
        for file in glob.glob(os.path.join(musan_path, '*/*/*/*.wav')):
            noise_type = file.split(os.sep)[-4]
            if noise_type not in self.noiselist:
                self.noiselist[noise_type] = []
            self.noiselist[noise_type].append(file)
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        # Load two segments from one utterance
        sample_id = self.sample_ids[index]
        audio = self.load_wav_split(self.subseg_info[sample_id])    # np.arr of shape (2, n_samples)
        audio = audio.astype(np.float32)
        
        # Choose two augmentation profiles for two segments
        augment_profiles = []
        for i in range(2):
            # rir: randomly select a row from rir_files and a random gain in [-7,3]
            rir_filts = random.choice(self.rir_files)
            rir_gains = np.random.uniform(-7, 3, 1)
            # MUSAN: randomly select a noise type, a noise file and a random snr
            noisecat = random.choice(self.noisetypes)
            noisefile = random.choice(self.noiselist[noisecat].copy())
            snr = [random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])]
            
            # Decide augmentation method based on probability
            p = random.random()
            if p < 0.25:  # Add RIR only
                augment_profiles.append({'rir_filt': rir_filts, 'rir_gain': rir_gains, 'add_noise': None, 'add_snr': None})
            elif p < 0.50:  # Add MUSAN noise only
                augment_profiles.append({'rir_filt': None, 'rir_gain': None, 'add_noise': noisefile, 'add_snr': snr})
            else:  # Add both
                augment_profiles.append({'rir_filt': rir_filts, 'rir_gain': rir_gains, 'add_noise': noisefile, 'add_snr': snr})
        
        # Apply augmentations
        audio_aug = []
        audio_aug.append(self.augment_wav(audio[0], self.obj_fs, augment_profiles[0]))  # Segment 0 with aug 0
        audio_aug.append(self.augment_wav(audio[1], self.obj_fs, augment_profiles[0]))  # Segment 1 with aug 0 (for AAT)
        audio_aug.append(self.augment_wav(audio[1], self.obj_fs, augment_profiles[1]))  # Segment 1 with aug 1
        audio_aug = np.concatenate(audio_aug, axis=0)  # Shape: (3, n_samples)
        audio_aug = torch.FloatTensor(audio_aug)

        # apply feature_extractor to augmented audio
        audio_aug_feats = torch.vmap(self.feature_extractor)(audio_aug.unsqueeze(1))  # mel feature of shape (3, n_frames, n_mels). n_frames is fixed for all segments due to fixed max_dur
        
        return audio_aug_feats
    
    def load_wav_split(self, sample_id_info):
        """Load two non-overlapping segments from a file.
        Args:
            sample_id_info: dict with keys 'file', 'start', 'stop'
        Returns:
            feats_arr: np.arr of shape (2, n_samples)
        
        """
        # load audio of current sample id
        wav_path, start_time, stop_time = sample_id_info['file'], sample_id_info['start'], sample_id_info['stop']
        wav_dat = self.wav_dat_dic[wav_path]
        audio = wav_dat[0, int(start_time * self.obj_fs):int(stop_time * self.obj_fs)].unsqueeze(0) # torch.Tensor of shape (1, n_samples)
        
        # repeat if audio is shorter than required length
        audio_len_max = int(self.max_dur * self.obj_fs)
        while audio.shape[1] < 2 * audio_len_max:
            audio = torch.cat([audio, audio], dim=1)
        
        # Select two non-overlapping segments
        ## Define the start frames of two segments
        randsize = int(audio.shape[1] - (audio_len_max * 2))
        startframe = [random.randint(0, randsize) for _ in range(2)]
        startframe.sort()
        startframe[1] += audio_len_max
        startframe = np.array(startframe)
        np.random.shuffle(startframe)
        ## concatenate two segments
        feats = []
        for asf in startframe:
            feats.append(audio[0, int(asf):int(asf + audio_len_max)].numpy())
        feats_arr = np.stack(feats, axis=0)
        
        return feats_arr
    
    def augment_wav(self, audio, audio_sr, augment):
        """Apply augmentation to audio
        Args:
            audio: np.arr of shape (n_samples,)
            audio_sr: int, sample rate of audio
            augment: dict with keys 'rir_filt', 'rir_gain', 'add_noise', 'add_snr'
        Returns:
            audio: np.arr of shape (1, n_samples)
        """
        if augment['rir_filt'] is not None:
            rir = np.multiply(augment['rir_filt'], pow(10, 0.1 * augment['rir_gain']))
            audio = signal.convolve(audio, rir, mode='full')[:len(audio)]
        
        if augment['add_noise'] is not None:
            noiseaudio, noisesr = self.load_wav_noise(augment['add_noise']) # of shape (1, n_samples)
            noiseaudio = signal.resample_poly(noiseaudio, audio_sr, noisesr, axis=1)
            noiseaudio = noiseaudio.astype(np.float32)
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
            noise = np.sqrt(10 ** ((clean_db - noise_db - augment['add_snr']) / 10)) * noiseaudio
            audio = audio + noise
        else:
            audio = np.expand_dims(audio, 0)
        
        return audio
    
    def load_wav_noise(self, filename):
        """Load noise audio file
        Args:
            filename: path of noise audio file
        Returns:
            feat: np.arr of shape (1, n_samples)
            sample_rate: int, sample rate of the audio
        """
        # load noise audio file
        audio, sample_rate = soundfile.read(filename)
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        
        # repeat if audio is shorter than required length
        audio_len_max = int(self.max_dur * sample_rate)    # keep consistent with segment length in load_wav_split
        while audio.shape[0] <= audio_len_max:
            audio = np.hstack([audio, audio])
        
        # select a segement of required length
        startframe = int(random.randint(0, audio.shape[0] - audio_len_max))
        feat = np.stack([audio[int(startframe):int(startframe + audio_len_max)]], axis=0)
        feat = audio[None, startframe:(startframe + audio_len_max)] # shape: (1, max_audio)
        
        return feat, sample_rate


class ContrastiveEvalDataset(Dataset):
    """
    Dataset for evaluation: extract clean embeddings for EER computation.
    """
    def __init__(self, subseg_json, feature_extractor, sample_keys):
        import json
        
        self.feature_extractor = feature_extractor
        self.obj_fs = self.feature_extractor.sample_rate
        self.sample_keys = sample_keys
        
        # Load sub-segments info
        with open(subseg_json, 'r') as f:
            self.subseg_info = json.load(f)        
        
        # Pre-load all audio files
        self.wav_paths = list(set([self.subseg_info[sid]['file'] for sid in self.sample_keys]))
        self.wav_dat_dic = {wav_path: load_audio(wav_path, self.obj_fs) for wav_path in self.wav_paths}
    
    def __len__(self):
        return len(self.sample_keys)
    
    def __getitem__(self, index):
        sample_id = self.sample_keys[index]
        sample_id_info = self.subseg_info[sample_id]
        # load audio of current sample id
        wav_path, start_time, stop_time = sample_id_info['file'], sample_id_info['start'], sample_id_info['stop']
        wav_dat = self.wav_dat_dic[wav_path]
        audio = wav_dat[0, int(start_time * self.obj_fs):int(stop_time * self.obj_fs)].unsqueeze(0) # torch.Tensor of shape (1, n_samples)
        audio_feat = torch.vmap(self.feature_extractor)(audio.unsqueeze(0)).squeeze(0)  # mel feature of shape (n_frames, n_mels)
        
        return audio_feat, sample_id


class AATNet(nn.Module):
    """
    Augmentation Adversarial Training discriminator.
    Distinguishes between same segment with different augmentations vs different segments.

    Input: concatenated embeddings of two segments (384-dim for 192-dim embeddings)

    Output: 2-class logits (same segment=1, different segment=0)

    NOTE: embedding_dim needs to adapt cam++
    """
    def __init__(self, embedding_dim=192, **kwargs):
        super(AATNet, self).__init__()
        layers = []
        layers.append(torch.nn.Sequential(
            nn.BatchNorm1d(embedding_dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(embedding_dim * 2, 512),
        ))
        layers.append(torch.nn.Sequential(
            nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 2),
        ))
        self.matcher = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.matcher(x)


class Reverse(nn.Module):
    """
    Gradient reversal layer for adversarial training.
    """
    def __init__(self, **kwargs):
        super(Reverse, self).__init__()
        self.matcher = torch.nn.Sequential(RevGrad())

    def forward(self, x):
        return self.matcher(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for self-supervised learning.
    This function is modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    """
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(ContrastiveLoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
    
    def forward(self, features):
        """
        Args:
            features: shape (batch_size, 2, embedding_dim)
                      features[:, 0, :] - embeddings from augmentation 1
                      features[:, 1, :] - embeddings from augmentation 2
        """
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
        count = features.shape[1]
        
        # Concatenate features
        feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # Compute cosine similarity
        dot_feature = F.cosine_similarity(feature.unsqueeze(-1), feature.unsqueeze(-1).transpose(0, 2))
        # Apply learned scaling
        torch.clamp(self.w, 1e-6)
        dot_feature = dot_feature * self.w + self.b
        # Numerical stability
        logits_max, _ = torch.max(dot_feature, dim=1, keepdim=True)
        logits = dot_feature - logits_max.detach()
        
        # Create masks
        mask = mask.repeat(count, count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * count).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        loss = -(mask * log_prob).sum(1) / mask.sum(1)
        loss = loss.view(count, batch_size).mean()
        
        # Compute accuracy
        n = batch_size * 2
        label = torch.from_numpy(np.asarray(list(range(batch_size - 1, batch_size * 2 - 1)) + list(range(0, batch_size)))).to(features.device)
        logits_flat = logits.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
        prec1, _ = self.contrastive_accuracy(logits_flat.detach().cpu(), label.detach().cpu(), topk=(1, 2))
        
        return loss, prec1

    def contrastive_accuracy(self, output, target, topk=(1,)):
        """Compute the training accuracy based on outputs and labels"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def evaluate_model(model, feature_extractor, subseg_json, testEER_file, device):
    """
    Evaluate model by computing EER and minDCF on test pairs.
    """
    # Read test pairs
    sim_label_list, key1_list, key2_list = read_testEER(testEER_file)
    # Get unique sample keys
    sample_keys = list(set(key1_list + key2_list))
    
    # Create evaluation dataset
    eval_dataset = ContrastiveEvalDataset(subseg_json, feature_extractor, sample_keys)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Extract embeddings
    model.eval()
    embeds_dict = {}
    
    with torch.no_grad():
        for feats, keys in eval_loader:
            feats = feats.to(device)
            embeddings = model(feats)
            embeddings = embeddings.cpu().numpy()
            for i, key in enumerate(keys):
                embeds_dict[key] = embeddings[i]
    
    # Compute scores
    eer, minDCF = evaluate_EER(embeds_dict, testEER_file)
    
    return eer, minDCF

class SimpleMeter:
    """Lightweight meter compatible with ProgressMeter usage in this file."""
    def __init__(self, name='', fmt=':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0.0

    def __str__(self):
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"

def train_one_epoch(train_loader, model, contrastive_criterion, aat_net, reverse_layer, 
                   optimizer_net, optimizer_aat, epoch, logger, device):
    """Train for one epoch with AAT framework"""
    batch_time = SimpleMeter('Time', fmt=':.3f')
    data_time = SimpleMeter('Data', fmt=':.3f')
    losses = SimpleMeter('Loss', fmt=':.6f')
    top1 = SimpleMeter('Acc', fmt=':.3f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch)
    )
    
    model.train()
    aat_net.train()
    reverse_layer.train()
    
    # AAT criterion for discriminator
    aat_criterion = nn.CrossEntropyLoss()
    
    end = time.time()
    
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # Reshape data and move to device
        data = data.to(device)  # shape (batch_size, 3, n_frames, n_mels)
        batch_size = data.shape[0]
        data_reshaped = data.view(batch_size * 3, data.shape[2], data.shape[3])  # (batch_size*3, n_frames, n_mels)
        
        # Extract embeddings and split
        embeddings = model(data_reshaped)  # (batch_size*3, embedding_dim)
        embeddings = embeddings.view(batch_size, 3, -1)
        feat_a = embeddings[:, 0, :]  # Segment 0 with aug 0
        feat_s = embeddings[:, 1, :]  # Segment 1 with aug 0
        feat_p = embeddings[:, 2, :]  # Segment 1 with aug 1
        
        # ===== Step 1: Train discriminator (AAT) =====
        # Detach encoder outputs to only train discriminator
        out_a = feat_a.detach()
        out_s = feat_s.detach()
        out_p = feat_p.detach()
        
        # Prepare inputs and labels for AAT
        ## Inputs: concatenated embeddings
        in_AAT = torch.cat([
            torch.cat((out_a, out_s), dim=1),  # Positive pairs (same aug, diff segment)
            torch.cat((out_a, out_p), dim=1)   # Negative pairs (diff aug, diff segment)
        ], dim=0)
        # Labels: [1,1,1,...,0,0,0,...]
        AAT_labels = torch.LongTensor([1] * batch_size + [0] * batch_size).to(device)
        
        # Forward through discriminator
        out_AAT = aat_net(in_AAT)
        dloss = aat_criterion(out_AAT, AAT_labels)
        # Backward and update discriminator
        optimizer_aat.zero_grad()
        dloss.backward()
        optimizer_aat.step()
        
        # ===== Step 2: Train encoder (adversarial + contrastive) =====
        optimizer_net.zero_grad()
        
        # Prepare inputs (now without detach to update encoder)
        in_AAT = torch.cat([
            torch.cat((feat_a, feat_s), dim=1),
            torch.cat((feat_a, feat_p), dim=1)
        ], dim=0)
        
        # Calculate losses
        ## Forward through gradient reversal and discriminator
        out_AAT = aat_net(reverse_layer(in_AAT))
        closs = aat_criterion(out_AAT, AAT_labels)
        ## Contrastive loss
        features = torch.stack([feat_a, feat_p], dim=1) # shape: (batch_size, 2, embedding_dim)
        sloss, prec1 = contrastive_criterion(features)
        ## Total loss: contrastive loss + AAT loss * weight
        nloss = sloss + closs * 3.0
        
        # Backward and update encoder
        nloss.backward()
        optimizer_net.step()
        
        # Update metrics
        losses.update(nloss.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 10 == 0:
            progress.display(i)
    
    return losses.avg, top1.avg


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
    
    # Create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    model_save_dir = os.path.join(args.result_dir, 'contrastive_models')
    os.makedirs(model_save_dir, exist_ok=True)
    # Setup logger
    logger = get_logger(os.path.join(args.result_dir, 'contrastive_training.log'))
    logger.info(f"Arguments: {args}")
    
    # Load configuration
    conf = yaml_config_loader(args.conf)
    conf_model = update_conf(conf, args.speaker_model_id, args.speaker_pretrained_model, rank)
    config_model = Config(conf_model)
    
    # Get feature extractor from config
    feature_extractor = build('feature_extractor', config_model)
    # Build speaker model and load pretrained weights
    logger.info("Loading speaker model...")
    embedding_dim = config_model.embedding_model["args"]["embedding_size"]
    speaker_model = build('embedding_model', config_model)
    pretrained_state = torch.load(config_model.pretrained_model, map_location='cpu')
    speaker_model.load_state_dict(pretrained_state)
    speaker_model.to(device)
    # Check if the device is GPU
    if device.type == 'cuda':
        print(f"[INFO]: Speaker model using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("[INFO]: Speaker model using CPU")

    # Prepare training dataset
    logger.info("Preparing training dataset...")
    train_dataset = ContrastiveTrainDataset(
        subseg_json = args.subseg_json,
        max_dur=args.max_dur,
        musan_path=args.musan_path,
        rir_filepath=args.rir_filepath,
        feature_extractor=feature_extractor
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=5
    )
    
    # Setup loss, AAT components and optimizers
    contrastive_criterion = ContrastiveLoss().to(device)
    aat_net = AATNet(embedding_dim).to(device)
    reverse_layer = Reverse().to(device)
    
    # Separate optimizers for encoder and discriminator
    optimizer_net = torch.optim.Adam(
        list(speaker_model.parameters()) + list(contrastive_criterion.parameters()),
        lr=args.lr
    )
    optimizer_aat = torch.optim.Adam(
        aat_net.parameters(),
        lr=args.lr
    )
    
    logger.info("AAT components initialized")
    logger.info(f"AATNet parameters: {sum(p.numel() for p in aat_net.parameters())/1e6:.2f}M")
    
    # Training loop
    logger.info("Starting contrastive learning training with AAT...")
    best_eer = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    score_file = open(os.path.join(args.result_dir, 'contrastive_scores.txt'), 'w')
    
    for epoch in range(1, args.max_epochs + 1):
        start_time_train = time.time()
        
        # Train for one epoch with AAT
        avg_loss, avg_acc = train_one_epoch(
            train_loader, speaker_model, contrastive_criterion, aat_net, reverse_layer,
            optimizer_net, optimizer_aat, epoch, logger, device
        )
        
        elapsed_time_train = time.time() - start_time_train
        
        # Evaluate
        if epoch % args.test_interval == 0:
            start_time_eval = time.time()
            
            eer, minDCF = evaluate_model(
                speaker_model, feature_extractor, args.subseg_json, 
                args.testEER_file, device)
            
            elapsed_time_eval = time.time() - start_time_eval
            
            log_msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}, "
                      f"Epoch {epoch}, LR {args.lr:.6f}, Acc {avg_acc:.2f}, "
                      f"LOSS {avg_loss:.6f}, EER {eer:.4f}, minDCF {minDCF:.3f}, "
                      f"Train Time {elapsed_time_train:.2f}s, Eval Time {elapsed_time_eval:.2f}s")
            logger.info(log_msg)
            score_file.write(log_msg + '\n')
            score_file.flush()
            
            # Save best model
            if eer < best_eer:
                # Remove old best model
                if best_epoch > 0:
                    old_model_path = os.path.join(model_save_dir, f'model_epoch_{best_epoch:04d}.pth')
                    if os.path.exists(old_model_path):
                        os.remove(old_model_path)
                
                # Save new best model
                best_eer = eer
                best_epoch = epoch
                patience_counter = 0
                
                model_path = os.path.join(model_save_dir, f'model_epoch_{epoch:04d}.pth')
                torch.save(speaker_model.state_dict(), model_path)
                logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.early_stop_patience:
                logger.info(f"Early stopping triggered. No improvement in EER for {args.early_stop_patience} epochs.")
                logger.info(f"Best EER: {best_eer:.4f} at epoch {best_epoch}")
                break
        
        else:
            log_msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}, "
                      f"Epoch {epoch}, LR {args.lr:.6f}, Acc {avg_acc:.2f}, "
                      f"LOSS {avg_loss:.6f}, Train Time {elapsed_time_train:.2f}s")
            logger.info(log_msg)
            score_file.write(log_msg + '\n')
            score_file.flush()
    
    score_file.close()
    logger.info(f"Training completed. Best EER: {best_eer:.4f} at epoch {best_epoch}")
    
    # Save final summary
    summary = {
        'best_epoch': best_epoch,
        'best_eer': best_eer,
        'total_epochs': epoch
    }
    
    with open(os.path.join(args.result_dir, 'contrastive_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
