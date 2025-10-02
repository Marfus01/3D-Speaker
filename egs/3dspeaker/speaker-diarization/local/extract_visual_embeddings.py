# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script uses pretrained models to perform speaker visual embeddings extracting.
This script use following open source models:
    1. Face detection: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
    2. Active speaker detection: TalkNet, https://github.com/TaoRuijie/TalkNet-ASD
    3. Face quality assessment: https://modelscope.cn/models/iic/cv_manual_face-quality-assessment_fqa
    4. Face recognition: https://modelscope.cn/models/iic/cv_ir101_facerecognition_cfglint
"""

import os
import sys
import json
import argparse

import torch
import torch.distributed as dist

current_file_path = os.path.abspath(__file__)
project_root = os.path.join(
    os.path.dirname(current_file_path),  # local/
    '..', '..', '..', '..'               # 回到 3D-Speaker/ 目录
)
project_root = os.path.abspath(project_root)
speakerlab_path = os.path.join(project_root, 'speakerlab')
assert os.path.exists(speakerlab_path), f"Cannot find speakerlab directory {speakerlab_path}. Please check the project root path."

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Project root added to sys.path: {project_root}")
    
from vision_processer import VisionProcesser
from speakerlab.utils.config import yaml_config_loader, Config

parser = argparse.ArgumentParser(description='Extract visual speaker embeddings for diarization.')
parser.add_argument('--conf', default=None, help='Config file')
parser.add_argument('--videos', default=None, help='Video list file')
parser.add_argument('--vad', default=None, help='Input vad info')
parser.add_argument('--onnx_dir', default='', type=str, help='Pretrained onnx directory')
parser.add_argument('--embs_out', default='', type=str, help='Out embedding dir')
parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')

# 将一组可能有重叠的时间区间合并为一组无重叠的区间，便于后续处理
def merge_overlap_region(vad_time_list):
    vad_time_list.sort(key=lambda x: x[0])
    out_vad_time_list = []
    for time in vad_time_list:
        if len(out_vad_time_list)==0 or time[0] > out_vad_time_list[-1][1]:
            out_vad_time_list.append(time)
        else:
            out_vad_time_list[-1][1] = time[1]
    return out_vad_time_list

def main():
    # Initialization
    args = parser.parse_args()
    conf = yaml_config_loader(args.conf)
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='gloo')

    if args.use_gpu:
        gpu_id = int(args.gpu[rank%len(args.gpu)])
        if gpu_id < torch.cuda.device_count():
            device = 'cuda'
        else:
            print("[WARNING]: Gpu %s is not available. Use cpu instead." % gpu_id)
            device = 'cpu'
    else:
        device = 'cpu'

    # obtain list of video filepaths
    with open(args.videos, 'r') as f:
        videos = [i.strip() for i in f.readlines()]

    videos.sort()
    if len(videos) == 0:
        raise Exception(
            "No videos found! Please check if the video file is accuratly generated."
            )
    if len(videos) <= rank:
        print("[WARNING]: The number of threads exceeds the number of files.")
        sys.exit()

    # 将 vad.json 的内容按所属video分组，整理为 dict 格式的 vad_data，key(str) 是video文件名，value(dict) 是从该video中提取的所有segment info(包含 id, start, stop, filepath)
    with open(args.vad, "r") as f:
        vad_json = json.load(f) # record the time regions which need to be extracted in each video

    vad_data={}
    for vpath in videos:
        rec_id = os.path.basename(vpath).rsplit('.', 1)[0]
        subset = {}
        for key in vad_json:
            k = str(key)
            if k.rsplit('_', 2)[0]==rec_id:
                subset[key] = vad_json[key]
        assert len(subset) > 0, "[ERROR] No vad info found in %s for %s." %(args.vad, vpath)
        vad_data[rec_id] = subset

    print("[INFO]: Start computing visual embeddings...")
    os.makedirs(args.embs_out, exist_ok=True)    
    local_videos = videos[rank::threads_num]  # video files to be processed by this thread

    for vpath in local_videos:
        # get filename without suffix of current video
        filename = os.path.basename(vpath)
        rec_id = filename.rsplit('.', 1)[0]
        # convert vad info to a list of sorted time regions without overlap 
        rec_vad_data = vad_data[rec_id] # segments' info
        rec_vad_time_list = [[v['start'], v['stop']] for v in rec_vad_data.values()]
        rec_vad_time_list = merge_overlap_region(rec_vad_time_list)
        # get corresponding audio path
        audio_path = os.path.join(os.path.dirname(vpath), '%s.wav'%rec_id)

        # extract visual embeddings in current video and save them
        embs_out_path = os.path.join(args.embs_out, '%s.pkl'%rec_id)
        if not os.path.isfile(embs_out_path):
            vprocesser = VisionProcesser(vpath, audio_path, rec_vad_time_list, embs_out_path, 
                                        args.onnx_dir, conf, device, gpu_id)
            vprocesser.run()
        else:
            print("[WARNING]: Embeddings has been saved previously. Skip it.")
        

if __name__ == '__main__':
    main()
