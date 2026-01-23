# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download the pretrained speaker embedding models from modelscope 
(https://www.modelscope.cn/models) based on the given model id, and extract speaker 
embeddings from subsegments of audio. Please pre-install "modelscope".
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np

import torch
import torchaudio
import torch.distributed as dist

current_file_path = os.path.abspath(__file__)
# 从'local/'回到'speaker-diarization'目录
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path),'..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from speakerlab.utils.config import yaml_config_loader, Config
from speakerlab.utils.builder import build
from speakerlab.utils.fileio import load_audio
from speakerlab.utils.utils import circle_pad

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

parser = argparse.ArgumentParser(description='Extract speaker embeddings for diarization.')
parser.add_argument('--model_id', default=None, help='Model id in modelscope')
parser.add_argument('--pretrained_model', default=None, type=str, help='Path of local pretrained model')
parser.add_argument('--conf', default=None, help='Config file')
parser.add_argument('--subseg_json', default='', type=str, help='Sub-segments info')
parser.add_argument('--embs_out', default='', type=str, help='Out embedding dir')
parser.add_argument('--batchsize', default=1, type=int, help='Batchsize for extracting embeddings')
parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')


# common settings for feature extractor
## assert sr=16k --> select single channel --> 提取形状为 [num_frames, n_mels] 的· --> 对每帧做均值归一化
## 梅尔谱特征的提取过程：将输入[1, num_samples]的音频信号分帧（每帧 25ms，帧移 10ms）--> 将每帧加窗后做FFT --> 计算功率谱 --> 通过80组梅尔滤波器 --> 对数压缩 --> 得到梅尔谱特征
FEATURE_COMMON = {
    'obj': 'speakerlab.process.processor.FBank',
    'args': {
        'n_mels': 80,
        'sample_rate': 16000, # 将所有音频都转换到16kHz
        'mean_nor': True,
    },
}

# 可用 speaker embedding model 的清单及基本配置
CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    'iic/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX,
        'model_pt': 'campplus_voxceleb.bin',
    },
    'iic/speech_campplus_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_VOX,
        'model_pt': 'campplus_cn_3dspeaker.bin',
    },
    'damo/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'damo/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_en_common.pt',
    },
}    

def update_conf(conf, model_id, pretrained_model, rank):
    """根据 model_id 或 pretrained_model 更新 conf 中的 model 和 pretrained_model 字段"""
    # !!! please set the correct feature extractor and model architecture !!! 
    conf['feature_extractor'] = FEATURE_COMMON
    # download the pretrained model(if necessary) and set the model config
    if model_id is not None:
        # obtain config of the selected model
        assert isinstance(model_id, str) and \
        is_official_hub_path(model_id), "Invalid modelscope model id."
        assert model_id in supports, "Model id not currently supported."
        model_config = supports[model_id]
        conf['embedding_model'] = model_config['model']
        if pretrained_model is None:
            # Check whether model files exist in local cache. If not, download from modelscope.
            if rank == 0:
                cache_dir = snapshot_download(
                            model_id,
                            revision=model_config['revision'],
                            )
                obj_list = [cache_dir]
            else:
                obj_list = [None]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                dist.broadcast_object_list(obj_list, 0)
            # set complete config according to the downloaded model
            cache_dir = obj_list[0]
            pretrained_model = os.path.join(cache_dir, model_config['model_pt'])
        conf['pretrained_model'] = pretrained_model
    else:
        assert pretrained_model is not None, \
            "[ERROR] One of the params `model_id` and `pretrained_model` must be set."
        # use the local pretrained model
        print("[INFO]: Use the local pretrained model %s" % pretrained_model)
        conf['pretrained_model'] = pretrained_model
        conf['embedding_model'] = CAMPPLUS_COMMON
    return conf

def main():
    # 初始化分布式计算环境
    args = parser.parse_args()
    conf = yaml_config_loader(args.conf)
    rank = int(os.environ['LOCAL_RANK'])  # 当前进程id
    threads_num = int(os.environ['WORLD_SIZE']) # 总进程数
    dist.init_process_group(backend='gloo')
    # 根据 model_id 或 pretrained_model 更新 conf 中的 model 和 pretrained_model 字段
    conf = update_conf(conf, args.model_id, args.pretrained_model, rank)
    
    # 将 subseg.json 的内容按录音文件分组，整理为 dict 格式的 metadata，key(str) 是录音文件名，value(dict) 是从该录音文件中提取的所有sub-segment info(包含 id, start, stop, filepath)
    with open(args.subseg_json, "r") as f:
        subseg_json = json.load(f)
    ## get unique wav filenames
    all_keys = subseg_json.keys() # list of all sub-segment ids, like "E01-152"
    A = [i.rsplit('-', 1)[0] for i in all_keys] # list of all wav filenames, like 'E01'
    all_rec_ids = list(set(A))
    all_rec_ids.sort()
    if len(all_rec_ids) == 0:
        print("[WARNING]:No recording IDs found! Please check if json file is accuratly generated.")
    if len(all_rec_ids) <= rank:
        print("[WARNING]: The number of threads exceeds the number of files.")
        sys.exit()
    ## group sub-segments by recording id
    metadata={}
    for rec_id in all_rec_ids:
        subset = {}
        for key in subseg_json:
            k = str(key)
            if k.rsplit('-', 1)[0]==rec_id:
                subset[key] = subseg_json[key]
        metadata[rec_id]=subset

    print("[INFO]: Start computing embeddings...")

    # set gpu_id for current process and device
    if args.use_gpu:
        gpu_id = int(args.gpu[rank%len(args.gpu)])
        if gpu_id < torch.cuda.device_count():
            device = torch.device('cuda:%d'%gpu_id)
        else:
            print("[WARNING]: Gpu %s is not available. Use cpu instead." % gpu_id)
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    # get objects of feature extractor and embedding model
    config = Config(conf)
    feature_extractor = build('feature_extractor', config)
    embedding_model = build('embedding_model', config)

    # load pretrained model
    pretrained_state = torch.load(config.pretrained_model, map_location='cpu')
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    embedding_model.to(device)
    # Check if the device is GPU
    if device.type == 'cuda':
        print(f"[INFO]: Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("[INFO]: Using CPU")

    # compute embeddings of sub-segments, and save embeddings belong to the same wav into one pkl file
    os.makedirs(args.embs_out, exist_ok=True)    
    local_rec_ids = all_rec_ids[rank::threads_num]  # 当前进程负责处理的wav list。例如['file1', 'file5', ...]    
    for rec_id in local_rec_ids:
        # Input: dict of sub-segments info for the current wav file
        meta = metadata[rec_id] 
        # Output: save embeddings of all sub-segments from the same wav into one pkl file
        emb_file_name = rec_id + ".pkl"
        stat_emb_file = os.path.join(args.embs_out, emb_file_name)
        if not os.path.isfile(stat_emb_file):
            ## load whole audio
            wav_path = meta[list(meta.keys())[0]]['file']
            obj_fs = feature_extractor.sample_rate
            wav = load_audio(wav_path, obj_fs=obj_fs) # torch.tensor, (num_channels, num_samples). num_channels=1 in fact, since load_audio averages all channels.

            ## split original audio into sub-segments, wavs of len num_subsegs, each elements is (1, num_samples_i))
            wavs = [wav[0, int(meta[i]['start']*obj_fs):int(meta[i]['stop']*obj_fs)].unsqueeze(0) for i in meta] # only use the first channel

            ## extract embeddings in batch
            embeddings = []
            with torch.no_grad():
                for wav in wavs:
                    wavs_batch = wav.unsqueeze(0).to(device) # (1, 1, num_samples_i)
                    feats_batch = torch.vmap(feature_extractor)(wavs_batch) # convert each segment to mel feature, [1, num_frames, n_mels]
                    embeddings_batch = embedding_model(feats_batch).cpu()
                    embeddings.append(embeddings_batch)
            embeddings = torch.cat(embeddings, dim=0).numpy()

            stat_obj = {
                'embeddings': embeddings, 
                'times': [[meta[i]['start'], meta[i]['stop']] for i in meta],
                'subseg_ids': [i for i in meta]
                }
            pickle.dump(stat_obj, open(stat_emb_file,'wb'))
        else:
            print("[WARNING]: Embeddings has been saved previously. Skip it.")

if __name__ == "__main__":
    main()
