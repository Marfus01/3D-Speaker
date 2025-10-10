# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script is designed to cluster speaker embeddings and generate RTTM result files as output.
"""

import os
import sys
import argparse
import pickle
import pathlib
import numpy as np
import copy

current_file_path = os.path.abspath(__file__)
# 从'local/'回到'speaker-diarization'目录
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path),'..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
import json

parser = argparse.ArgumentParser(description='Cluster embeddings and output rttm files')
parser.add_argument('--conf', default=None, help='Config file')
parser.add_argument('--wavs', default=None, help='Wav list file')
parser.add_argument('--cluster_type', default='audio_only', type=str, help='Clustering type, support "audio_only" and "audio_vision"')
parser.add_argument('--audio_embs_dir', default=None, type=str, help='Embedding dir')
parser.add_argument('--result_dir', default=None, type=str, help='Result dir')
parser.add_argument('--visual_embs_dir', default=None, type=str, help='Visual embedding dir')

def make_rttms(seg_list, out_rttm, rec_id):
    """
    Generate an RTTM (Rich Transcription Time Marked) file for speaker diarization results.

    This function processes a list of speaker segments, merges overlapping or adjacent segments
    with the same speaker ID, and writes the processed segments to an RTTM file. The RTTM file
    is a standard format used for speaker diarization output.

    Args:
      seg_list (list of tuples): A list of tuples where each tuple contains:
        - A pair of floats (start_time, end_time) representing the time range of the segment.
        - An integer representing the speaker ID for the segment.
      out_rttm (str): The output file path where the RTTM file will be saved.
      rec_id (str): The recording ID to be included in the RTTM file.

    Output:
      Writes the processed speaker segments to the specified RTTM file in the following format:
      SPEAKER <rec_id> 0 <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>

    Example:
      seg_list = [((0.0, 1.5), 0), ((1.5, 3.0), 0), ((3.0, 4.0), 1)]
      out_rttm = "output.rttm"
      rec_id = "example_audio"
      make_rttms(seg_list, out_rttm, rec_id)
    """
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        # extract start time, end time, and speaker ID from the segment
        seg_st, seg_ed = seg[0]
        seg_st = float(seg_st)
        seg_ed = float(seg_ed)
        cluster_id = seg[1] + 1
        if i == 0:
            new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])
        # merge segments with the same speaker ID if they are overlapping
        elif cluster_id == new_seg_list[-1][3]:
            if seg_st > new_seg_list[-1][2]:
                new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][2] = seg_ed
        # if the speaker ID is different, check for overlap and adjust accordingly
        else:
            if seg_st < new_seg_list[-1][2]:
                p = (new_seg_list[-1][2]+seg_st) / 2
                new_seg_list[-1][2] = p
                seg_st = p
            new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])

    # write the processed segments to the RTTM file
    line_str ="SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
    with open(out_rttm,'w') as f:
        for seg in new_seg_list:
            seg_id, seg_st, seg_ed, cluster_id = seg
            f.write(line_str.format(seg_id, seg_st, seg_ed-seg_st, cluster_id))


def summary_cluster_results(labels, modal_type='audio'):
    """
    Summary statistics of cluster sizes
    """
    uniq = np.unique(labels)
    # Count occurrences of each unique label
    cluster_sizes = {label: np.sum(labels == label) for label in uniq}
    # Sort labels by their occurrence counts in descending order
    sorted_label_counts = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

    # Print the top 20 labels with their counts
    print("[INFO] Top 20 label counts:")
    for i, (label, count) in enumerate(sorted_label_counts[:20]):
        print(f"{modal_type} cluster {label}: of size {count};")

    # Aggregate counts for the remaining labels
    remaining_counts = {}
    for label, count in sorted_label_counts[20:]:
        if count not in remaining_counts:
            remaining_counts[count] = 0
        remaining_counts[count] += 1

    # Print the aggregated counts for remaining labels
    if len(remaining_counts) > 0:
        print("[INFO] Aggregated counts for remaining labels:")
        for count, num_labels in sorted(remaining_counts.items()):
            print(f"{num_labels} {modal_type} clusters of size {count}.")    


def save_cluster_results_audio(labels, audio_seg_ids, out_json):
    """
    Save clustering results to a JSON file.

    Args:
        labels (ndarray): Cluster labels for each embedding, of shape [N].
        audio_seg_ids (ndarray): Segment IDs corresponding to each embedding, of shape [N].
        out_json (str): Path to the output JSON file.

    Output:
        Saves a JSON file where each segment ID is a key and its corresponding cluster label is the value.
    """
    # Create a dictionary mapping segment IDs to cluster labels
    cluster_results = {seg_id: int(label) for seg_id, label in zip(audio_seg_ids, labels)}

    # Save the dictionary to a JSON file
    with open(out_json, 'w') as f:
        json.dump(cluster_results, f, indent=2)


def audio_only_func(local_wav_list, audio_embs_dir, result_dir, config):
    """
    仅有speaker embeddings时，通过SpectralCluster(会自动估计说话人数)-->将极小簇就近合并到较大簇-->根据聚类中心余弦相似度合并相似簇 的方式进行聚类
    """
    embeddings = np.array([], dtype=np.float32)
    audio_seg_ids = np.array([], dtype='<U50')   
    # 对每一个音频文件
    for file_idx, wav_file in enumerate(local_wav_list):
        # 加载前序步骤从当前wav中提取的所有speaker embeddings
        wav_name = os.path.basename(wav_file)
        rec_id = wav_name.rsplit('.', 1)[0]
        embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')
        if not os.path.exists(embs_file):
            print("[WARNING]: %s does not exist, it is possible that vad model did not detect valid speech in file %s, please check it."%(embs_file, wav_file))
            continue
        with open(embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            if file_idx == 0:                
                embeddings = stat_obj['embeddings']
                audio_seg_ids = stat_obj['subseg_ids']
            else:
                embeddings = np.vstack((embeddings, stat_obj['embeddings']))
                audio_seg_ids = np.hstack((audio_seg_ids, stat_obj['subseg_ids']))

    # cluster
    # pval_list = [config.cluster['args']['pval']*pow(0.5, i) for i in range(3)]
    pval_list = [0.012, 0.006, 0.004, 0.002]
    for pval in pval_list:
        print(f"[INFO] Updated 'pval' in cluster args to {pval}")
        config_copy = copy.deepcopy(config)
        config_copy.cluster['args']['pval'] = pval
        cluster = build('cluster', config_copy)
        labels = cluster(embeddings)
        summary_cluster_results(labels, modal_type='audio')
        out_json = os.path.join(result_dir, f'cluster_results_audio(pval={pval}, mer_cos=0.8).json')
        save_cluster_results_audio(labels, audio_seg_ids, out_json)

def audio_vision_func(local_wav_list, audio_embs_dir, visual_embs_dir, result_dir, config):
    # 对每一个音频文件，加载其对应的音频和视觉speaker embeddings，然后进行多模态聚类
    for wav_file in local_wav_list:
        wav_name = os.path.basename(wav_file)
        rec_id = wav_name.rsplit('.', 1)[0]
        audio_embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')
        visual_embs_file = os.path.join(visual_embs_dir, rec_id + '.pkl')
        with open(audio_embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            audio_embeddings = stat_obj['embeddings']
            audio_times = stat_obj['times']
        with open(visual_embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            visual_embeddings = stat_obj['embeddings']
            visual_times = stat_obj['times']

        # cluster
        ## 1. audio-only clustering: 仍使用谱聚类实现，聚类整体流程与audio_only_func中的描述相同。min_cluster_size和pval与只有语音模态时有所不同
        ## 2. visual-only clustering: 通过AHCluster(根据余弦相似度定义距离，根据提前定义的距离阈值，做层次聚类)-->将极小簇就近合并到较大簇-->根据聚类中心余弦相似度合并相似簇 的方式进行聚类
        ## 3. 设置visual簇从max_audio_spk_id开始编号，筛选至少一个visual segment与某audio簇的重叠时长>1s 的visual簇（以及与其overlap的audio segment embedding 的均值作为聚类中心），随后对于各个 audio 簇，查找与其重叠时长>0.5s的 visual 簇，并计算前者中各个样本与后者中各个聚类中心的余弦相似度，据此将所有audio segment分配到与其最相似的visual簇上（如果没有任何visual簇与其重叠>0.5s，则保持其audio-only聚类结果不变）。由于>0.5s的阈值并不苛刻，因此相当于利用visual信息重新分配了大部分audio segment的簇ID
        cluster = build('cluster', config)        
        labels = cluster(audio_embeddings, visual_embeddings, audio_times, visual_times, config)
        # output rttm
        new_labels = np.zeros(len(labels), dtype=int)
        uniq = np.unique(labels)
        for i in range(len(uniq)):
            new_labels[labels==uniq[i]] = i 
        seg_list = [(i,j) for i, j in zip(audio_times, new_labels)]
        out_rttm = os.path.join(result_dir, rec_id+'.rttm')
        make_rttms(seg_list, out_rttm, rec_id)

def main():
    args = parser.parse_args()
    # 获取所有待处理的wav文件列表
    with open(args.wavs,'r') as f:
        wav_list = [i.strip() for i in f.readlines()]
    wav_list.sort()

    os.makedirs(args.result_dir, exist_ok=True)
    print("[INFO] Start clustering...")
    # 加载yaml文件
    config = build_config(args.conf)
    assert args.cluster_type in ['audio_only', 'audio_vision'], f'--cluster_type should be either "audio_only" or "audio_vision", but got {args.cluster_type}'
    if args.cluster_type == 'audio_only':
        if hasattr(config, 'audio_cluster') and hasattr(config, 'vision_cluster'):
            config.cluster = config.audio_cluster
            del config.audio_cluster, config.vision_cluster
        audio_only_func(wav_list, args.audio_embs_dir, args.result_dir, config)
    else:
        assert args.visual_embs_dir is not None and args.visual_embs_dir != '', f'--visual_embs_dir should be provided when --cluster_type is "audio_vision"'
        audio_vision_func(wav_list, args.audio_embs_dir, args.visual_embs_dir, args.result_dir, config)


if __name__ == "__main__":
    main()
