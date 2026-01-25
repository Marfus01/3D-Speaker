# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download the pretrained models from pyannote/segmentation-3.0 (https://huggingface.co/pyannote/segmentation-3.0)
and perform the overlap detection given the audio. Please pre-install "pyannote".
"""

import os
import sys
import json
import numpy as np
import pickle
import argparse
import torch

try:
    import pyannote
except ImportError:
    raise ImportError("Package \"pyannote\" not found. Please install them first.")

from pyannote.audio import Inference, Model

parser = argparse.ArgumentParser(description='Overlap detection')
parser.add_argument('--wavs', default='', type=str, help='Wavs')
parser.add_argument('--out_dir', default='', type=str, help='output dir')
parser.add_argument('--hf_access_token', default='', type=str, help='hf_access_token for pyannote/segmentation-3.0')

def main():
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 将 wav.list中的所有内容读入为list
    wavs = []
    if args.wavs.endswith('.wav'):
        # input is a wav path
        wavs.append(args.wavs)
    else:
        try:
            # input is wav list
            with open(args.wavs,'r') as f:
                wav_list = f.readlines()
        except:
            raise Exception('Input should be a wav file or a wav list.')
        for wav_path in wav_list:
            wav_path = wav_path.strip()
            wavs.append(wav_path)
    # setting
    segmentation_params = {
        'segmentation':'pyannote/segmentation-3.0',
        'segmentation_batch_size':32,
        'use_auth_token':args.hf_access_token,
    }

    # 加载预训练模型，并将其初始化为Inference对象。pyannote/segmentation-3.0的输入为处理10秒单声道16kHz音频，输出为帧级别的包含非语音、3个独立说话人及3种重叠说话人组合的7分类矩阵。参考https://blog.csdn.net/gitblog_02861/article/details/149852839
    model = Model.from_pretrained(
        segmentation_params['segmentation'], 
        use_auth_token=segmentation_params['use_auth_token'], 
        strict=False,
    )

    _segmentation = Inference(
        model,
        duration=model.specifications.duration, # size of chunk to be processed, default 10.0s
        step=0.1 * model.specifications.duration,   # step size between two consecutive chunks, default 1.0s
        skip_aggregation=True,  # do not aggregate results over time windows
        batch_size=segmentation_params['segmentation_batch_size'],
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
    )

    def get_valid_field(count):
        """
        Get valid speech field from count results.
        Args:
            count: pyannote.core.SlidingWindowFeature, [total_frames_num, 1], count of speakers per frame
        Returns:
            valid_field: list of [start_time, end_time], valid speech segments
        """
        valid_field = []
        start = None    # start time of valid speech segment
        # for each frame
        for i, (c, data) in enumerate(count):
            if data.item()==0 or i==len(count)-1:   # non-speech
                if start is not None:
                    end = c.middle  # end time of valid speech segment
                    valid_field.append([start, end])
                    start = None
            else:
                if start is None:
                    start = c.middle
        return valid_field
    
    # 遍历处理每一个wav文件
    segmentation_dict = {}
    for wpath in wavs:
        basename = os.path.basename(wpath).rsplit('.', 1)[0]
        # segmentations: [chunk_num, frames_num, speakers_num]
        segmentations = _segmentation({'audio':wpath})  # (bs, n_frames, n_classes). each frame in audio is about 16ms
        frame_windows = _segmentation.model.receptive_field
        # count: [total_frames_num, 1]
        count = Inference.aggregate(
            np.sum(segmentations, axis=-1, keepdims=True),
            frame_windows,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )   # Aggregate all chunks into the full audio frame-level scores to get count of speakers per frame

        count.data = np.rint(count.data).astype(np.uint8)   # Round to integer
        valid_field = get_valid_field(count)
        segmentation_dict[basename] = {'segmentations': segmentations.data, 'count': count.data, 'valid_field': valid_field}    # store results: chunk-level predictions, frame-level count, valid speech segments
    out_path = os.path.join(args.out_dir, 'segmentation.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(segmentation_dict, f)

    msg = "Segmentation file is prepared in %s" % (out_path)
    print(f'[INFO]: {msg}')

if __name__ == '__main__':
    main()
