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
import copy, time
from hmmlearn import hmm

current_file_path = os.path.abspath(__file__)
# 从'local/'回到'speaker-diarization'目录
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path),'..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.process.cluster import summary_cluster_results, reset_cluster_ids, align_clusters2clusters, align_samples2clusters, get_unreliable_metrics
import json
from datetime import datetime
from speakerlab.process.hmm.hmm_X import HMM_X
from speakerlab.process.hmm.nested_hmm import NestedHMM
from speakerlab.process.hmm.nested_hmm_full import NestedHMM_full

parser = argparse.ArgumentParser(description='Cluster embeddings and output rttm files')
parser.add_argument('--conf', default=None, help='Config file')
parser.add_argument('--wavs', default=None, help='Wav list file')
parser.add_argument('--cluster_type', default='audio_only', type=str, help='Clustering type, support "audio_only" and "audio_vision"')
parser.add_argument('--visual_info_type', default='vad+key_frame', type=str, help='Visual information type, support "vad", "key_frame", "vad+key_frame"')
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

def count_consecutive_segment_lengths(arr, seg_lengths=None):
    """
    Given a 1D numpy array of integers, returns an array of the same shape where each element
    is replaced by the length of the consecutive segment it belongs to.

    Example:
        [0,0,2,3,3,3,4] --> [2,2,1,3,3,3,1]
    """
    if seg_lengths is None:
        seg_lengths = np.array([len(arr)])
    arr = np.asarray(arr)
    lengths = np.array([], dtype=int)

    seg_start = 0
    for seg_len in seg_lengths:
        seg_end = seg_start + seg_len
        seg_arr = arr[seg_start:seg_end]
        # 找到分段的起始位置
        boundaries = np.flatnonzero(np.diff(seg_arr)) + 1
        # 在首尾补上0和len(arr)
        boundaries = np.concatenate(([0], boundaries, [len(seg_arr)]))
        # 计算每段长度
        seg_lengths_inner = np.diff(boundaries)   # like [2,1,3,1]
        lengths = np.concatenate((lengths, seg_lengths_inner))
        seg_start = seg_end

    return lengths

def count_consecutive_ones(arr, seg_lengths=None):
    arr = np.asarray(arr)
    if seg_lengths is None:
        seg_lengths = np.array([len(arr)])
    arr = np.asarray(arr)
    lengths = np.array([], dtype=int)

    seg_start = 0
    for seg_len in seg_lengths:
        seg_end = seg_start + seg_len
        seg_arr = arr[seg_start:seg_end]
        
        # 找到从0到1的起始位置和从1到0的结束位置
        padded = np.concatenate(([0], seg_arr, [0]))
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths = np.concatenate((lengths, ends - starts))

        seg_start = seg_end

    return lengths   

def alabels_hmmX_smooth(S_hat_onehot, X_onehot, lengths, audio_seg_ids, result_dir, flag_has_neg1=False,
                        selective_change=False, duration_dat= None):
    n_actors = S_hat_onehot.shape[1]    
    alabels = np.argmax(S_hat_onehot, axis=1)
    
    print("\n=== 训练模型 ===")
    start_time = time.time()
    print("训练开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    model = HMM_X(n_actors=n_actors, n_iter=100, tol=1e-3, verbose=True)
    model.fit(S_hat_onehot, X_onehot, lengths)
    end_time = time.time()
    print("训练结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print("训练耗时:", end_time - start_time, "秒")

    print("\n=== 模型参数 ===")
    print("说话人初始概率 β 的logits：\n", model.beta_)
    print("说话人转移矩阵 A_S_ 的logits：\n", model.A_S_)
    print("说话人识别混淆矩阵 B_S :\n", model.B_S_)
    print("协变量X取值为1对说话人初始状态的影响 η1_ :\n", model.eta1_)
    print("协变量X取值为1对说话人转移的影响 η2_ :\n", model.eta2_)

    # 使用训练好的模型解码隐藏状态及其后验概率
    pred_probs = model.predict_proba(S_hat_onehot, X_onehot, lengths)['speaker_states'] # 计算后验概率  (n_samples, n_states_hid)
    speaker_states_viterbi = model.predict(S_hat_onehot, X_onehot, lengths) # viterbi 解码结果
    speaker_states_viterbi_prob = np.array([pred_probs[i, state] for i, state in enumerate(speaker_states_viterbi)])
    if flag_has_neg1:  # 将-1标签还原回来
        speaker_states_viterbi[speaker_states_viterbi == n_actors - 1] = -1 
        alabels[alabels == n_actors - 1] = -1

    print("解码结果相较观测改变数量:", np.sum(alabels != speaker_states_viterbi))
    alabels_smoothed = copy.deepcopy(speaker_states_viterbi)
    smoothed_cluster_dic = {seg_id: int(label) for seg_id, label in zip(audio_seg_ids, alabels_smoothed)}
    with open(os.path.join(result_dir, 'cluster_results_audio_vision_vad_hmmx.json'), 'w', encoding='utf-8') as f:
        json.dump(smoothed_cluster_dic, f, indent=2)

    if selective_change == True:
        change_flags = alabels != speaker_states_viterbi
        uniq_lengths_change, lengths_counts_change = np.unique(count_consecutive_ones(change_flags, lengths), return_counts=True)
        print("解码序列相较观测连续改变次数统计:", dict(zip(uniq_lengths_change, lengths_counts_change)))
        change_lengths = count_consecutive_segment_lengths(change_flags, lengths)
        change_flags = np.repeat(change_lengths, change_lengths) * change_flags # 将与原始观测相同位置处的取值设为0
        selected_indices = np.where(change_flags == 1)[0]

        prop_keep=0.25
        pp_keep =int(prop_keep*100)
        num_keep = int(len(selected_indices) * prop_keep)
        selected_indices_keeped = selected_indices[np.argsort(speaker_states_viterbi_prob[selected_indices])[-num_keep:]]

        assert duration_dat is not None, "duration_dat should be provided when selective_change is True."
        bins = [0, 1, 2, 3, 4, float('inf')]
        duration_dat_groups = np.digitize(duration_dat[selected_indices_keeped], bins) - 1
        uniq_dur, dur_counts = np.unique(duration_dat_groups, return_counts=True)
        print(f"duration statistics for changed segments(under cond2, keep: {pp_keep}%):", dict(zip(uniq_dur, dur_counts)))
        selected_indices_keeped = [i for i in selected_indices_keeped if duration_dat[i] <= 1]

        alabels_smoothed = copy.deepcopy(alabels)
        alabels_smoothed[selected_indices_keeped] = speaker_states_viterbi[selected_indices_keeped]
        print(f"解码结果相较观测改变数量(cond2约束下, 选取top-{pp_keep}%): ", num_keep)

        smoothed_cluster_dic = {seg_id: int(label) for seg_id, label in zip(audio_seg_ids, alabels_smoothed)}
        with open(os.path.join(result_dir, f'cluster_results_audio_vision_vad_hmmx_cond(top-{pp_keep}%).json'), 'w', encoding='utf-8') as f:
            json.dump(smoothed_cluster_dic, f, indent=2)

def labels_nested_hmm_full_smooth(S_hat_onehot, F_hat, X_onehot, lengths, params,
                                  audio_seg_ids, result_dir, flag_has_neg1=False, alabels_unreliable_metrics=None, B_S_diag_min=None):
    n_actors = S_hat_onehot.shape[1]    
    alabels = np.argmax(S_hat_onehot, axis=1)
    print(f"Count of each actor in S_hat_onehot: {np.sum(S_hat_onehot, axis=0)}")
    print(f"Count of each actor in F_hat: {np.sum(F_hat, axis=0)}")
    print(f"Count of each actor in X_onehot: {np.sum(X_onehot, axis=0)}")
    
    model = HMM_X(n_actors=n_actors, n_iter=100, tol=1e-3, verbose=True, params=params)
    print(f"\n=== 训练模型(covariate_mode = {model.covariate_mode}) ===")
    if B_S_diag_min is not None:
        print(f"说话人识别混淆矩阵 B_S 的对角线最小值: {B_S_diag_min}") 
    start_time = time.time()
    print("训练开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    model.fit(S_hat_onehot, X_onehot, F_hat, B_S_diag_min, lengths)
    end_time = time.time()
    print("训练结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print("训练耗时:", end_time - start_time, "秒")

    print("\n=== 模型参数 ===")
    print("说话人初始概率 β 的logits：\n", model.beta_)
    print("说话人转移矩阵 A_S_ 的logits：\n", model.A_S_)
    print("说话人识别混淆矩阵 B_S :\n", model.B_S_)
    if model.covariate_mode in ['X_only', 'both']:
        print("协变量X取值为1对说话人初始状态的影响 η1_ :\n", model.eta1_)
        print("协变量X取值为1对说话人转移的影响 η2_ :\n", model.eta2_)
    if model.covariate_mode in ['F_only', 'both']:
        print("中间帧出现某个角色的人脸对说话人初始状态的影响 γ₁_ :\n", model.gamma1_)
        print("中间帧出现某个角色的人脸对说话人转移的影响 γ₂_ :\n", model.gamma2_)

    # 使用训练好的模型解码隐藏状态
    speaker_states_viterbi = model.predict(S_hat_onehot, X_onehot, F_hat, lengths) # viterbi 解码结果

    save_name_part_has_neg1 ='_has_neg1' if flag_has_neg1 else ''
    save_name_part_B_S = f'_B_S_diagmin={B_S_diag_min}' if B_S_diag_min is not None else ''
    
    if flag_has_neg1:  # 将说话人的-1标签还原回来
        speaker_states_viterbi[speaker_states_viterbi == n_actors - 1] = -1 
        alabels[alabels == n_actors - 1] = -1

    print("说话人解码结果相较观测改变数量:", np.sum(alabels != speaker_states_viterbi))
    if alabels_unreliable_metrics is not None:
        for unreliable_pp in [2, 5, 10, 15, 20, 35, 50, 75, 100]:
            changed_idxs = np.argsort(alabels_unreliable_metrics)[:int(unreliable_pp / 100 * len(alabels))] # indexs of elements in smallest alabels_unreliable_metrics
            alabels_smoothed = copy.deepcopy(alabels)
            alabels_smoothed[changed_idxs] = speaker_states_viterbi[changed_idxs]
            print(f"unreliable_percent={unreliable_pp}时，选择性平滑结果相较观测改变数量:", np.sum(alabels != speaker_states_viterbi))
            smoothed_cluster_dic = {seg_id: int(label) for seg_id, label in zip(audio_seg_ids, alabels_smoothed)}
            with open(os.path.join(result_dir, f'cluster_results_audio_vision_vad_hmmx_{model.covariate_mode}_{save_name_part_B_S}{save_name_part_has_neg1}(unreliable_pp={unreliable_pp}).json'), 'w', encoding='utf-8') as f:
                json.dump(smoothed_cluster_dic, f, indent=2)

    

def extract_aligned_vlabels_results(vlabels, vlabels_aligned_dic, visual_times=None):
    """
    Filter and align vlabels and visual_times(optimal) based on a mapping dictionary.

    This function filters the vlabels  and visual_times arrays to include only the rows where
    the vlabels values are present as keys in the vlabels_aligned_dic. It then maps the filtered
    vlabels to new labels using the dictionary.

    Args:
        vlabels (np.array): A 1D numpy array of length n, representing visual cluster labels.
        vlabels_aligned_dic (dict): A dictionary where keys are a subset of vlabels, and values
                                    are the corresponding new labels.
        visual_times (np.array, optimal): A 1D numpy array of length n, representing visual segment times.

    Returns:
        np.array: A 1D numpy array of length m, mapped vlabels_new.
        np.array: A 1D numpy array of length m, filtered visual_times.
        np.array: A boolean mask array of length n, indicating which rows were kept.
    """
    # Filter rows where vlabels are in vlabels_aligned_dic keys or -1（for mid frame part）
    mask = np.isin(vlabels, list(vlabels_aligned_dic.keys())) | (vlabels == -1)
    filtered_vlabels = vlabels[mask]
    # Map filtered vlabels to new labels using the dictionary
    vlabels_new = np.array([vlabels_aligned_dic[label] if label in vlabels_aligned_dic else -1 for label in filtered_vlabels])

    if visual_times is not None:
        filtered_visual_times = visual_times[mask]
    else:
        filtered_visual_times = None
    return vlabels_new, filtered_visual_times, mask

def process_top_cluster_ids_together(alabels, vlabels_vad_aligned, vlabels_mf_aligned=None, main_actors_num=1):
    """
    Reset cluster IDs to be consecutive integers starting from -1, while retaining only the top clusters.

    This function processes 3 sets of cluster labels (`alabels`, `vlabels_vad_aligned` and `vlabels_mf_aligned`) and ensures that only the top clusters (based on size) are retained. The remaining clusters are assigned a label of -1.

      alabels (ndarray): Array of original cluster labels for audio, of shape [N_a].
      vlabels_vad_aligned (ndarray): Array of aligned cluster labels for vad face, of shape [N_vv].
      vlabels_mf_aligned (ndarray, optional): Array of aligned cluster labels for mid frame face, of shape [N_vm].
      main_actors_num (int, optional): The number of main actors to retain. Only the top `2 * main_actors_num` clusters (based on size) will be retained. Defaults to 1.

      tuple: A tuple containing:
        - new_alabels (ndarray): Updated cluster labels for audio, with non-top clusters set to -1.
        - new_vlabels_vad_aligned (ndarray): Updated cluster labels for vad face, with non-top clusters set to -1.
        - new_vlabels_mf_aligned (ndarray or None): Updated cluster labels for mid frame face, with non-top clusters set to -1. Returns None if `vlabels_mf_aligned` is not provided.

    Raises:
      AssertionError: If `vlabels_aligned` contains labels not present in `alabels`.
    """
    uniq_a = np.unique(alabels)
    if vlabels_mf_aligned is None:
        uniq_v = np.unique(vlabels_vad_aligned)
    else:
        assert (set(vlabels_mf_aligned) - set([-1])).issubset(set(vlabels_vad_aligned)), "vlabels_mf_aligned contains labels not present in alabels, and they are not -1."
        uniq_v = np.unique(np.concatenate((vlabels_vad_aligned, vlabels_mf_aligned)))
    assert (set(uniq_v)- set([-1])).issubset(set(uniq_a)), "vlabels_aligned contains labels not present in alabels: {}".format(set(uniq_v) - set([-1]) - set(uniq_a))

    # Count occurrences of each unique alabel
    uniq_a_count = {aid: np.sum(alabels == aid) for aid in uniq_a}
    # Sort alabels by count (descending), then by audio cluster id value (descending)
    sorted_uniq_a = sorted(uniq_a_count.keys(), key=lambda x: (-uniq_a_count[x], -x))

    new_alabels = np.full(len(alabels), -1, dtype=int)  # Default all alabels to -1
    new_vlabels_vad_aligned = np.full(len(vlabels_vad_aligned), -1, dtype=int)  # Default all vlabels_vad_aligned to -1
    new_vlabels_mf_aligned = None
    if vlabels_mf_aligned is not None:
        new_vlabels_mf_aligned = np.full(len(vlabels_mf_aligned), -1, dtype=int)  # Default all vlabels_mf_aligned to -1
        
    # Retain only the top 2 * main_actors_num clusters
    top_clusters = sorted_uniq_a[:2 * main_actors_num]
    for new_id, old_id in enumerate(top_clusters):
        new_alabels[alabels == old_id] = new_id
        new_vlabels_vad_aligned[vlabels_vad_aligned == old_id] = new_id
        if vlabels_mf_aligned is not None:
            new_vlabels_mf_aligned[vlabels_mf_aligned == old_id] = new_id

    return new_alabels, new_vlabels_vad_aligned, new_vlabels_mf_aligned

def convert_alabels_to_onehot(audio_seg_ids, alabels, ncols):
    """
    Converts audio labels into a one-hot encoded matrix.

    Args:
      audio_seg_ids (np.ndarray): A 1D array of shape (N,) containing unique identifiers for each audio segment.
      alabels (np.ndarray): A 1D array of shape (N,) containing the labels for each audio segment. Labels are integers starting from -1 or 0, where -1 indicates 'others'.

    Returns:
      np.ndarray: A 2D one-hot encoded matrix of shape (N, K+2) or (N, K+1), where K is the maximum label in `alabels`. The last column corresponds to the -1 label if present.
    """
    assert len(audio_seg_ids) == len(alabels), f"audio_seg_ids and alabels must have the same length, but got {len(audio_seg_ids)} and {len(alabels)}."
    n_major_clusters = np.max(alabels) + 1
    if -1 in alabels:
        assert set(np.unique(alabels)) == set(range(-1, n_major_clusters)), "alabels contains non-consecutive integers starting from -1."
    else:
        assert set(np.unique(alabels)) == set(range(0, n_major_clusters)), "alabels contains non-consecutive integers starting from 0."
    S_hat_onehot = np.zeros((len(audio_seg_ids), ncols), dtype=int)  # last column for -1 label if present

    for idx, label in enumerate(alabels):
        if label == -1:
            S_hat_onehot[idx, -1] = 1
        else:
            S_hat_onehot[idx, label] = 1

    return S_hat_onehot


def convert_vlabels_vad_to_onehot(audio_times, visual_times_vad, audio_seg_ids, vlabels_vad, S_hat_onehot):
    """
    Converts visual labels of active speaker face into a one-hot encoded matrix mapped to audio segments.

    Args:
      audio_times (np.ndarray): A 2D array of shape (N, 2) where each row represents the start and end times of an audio segment.
      visual_times_vad (np.ndarray): A 1D array of shape (M,) where each element represents the timestamp of a active speaker face.
      audio_seg_ids (np.ndarray): A 1D array of shape (N,) containing unique identifiers for each audio segment.
      vlabels_vad (np.ndarray): A 1D array of shape (M,) containing the labels for each active speaker face. Labels are integers starting from -1 or 0, where -1 indicates 'others'.
      S_hat_onehot (np.ndarray): The one-hot encoded matrix of shape (N, p) for audio labels, used to determine the shape.

    Returns:
      np.ndarray: A 2D one-hot encoded matrix of the same shape as `S_hat_onehot`, where visual labels are mapped to their corresponding audio segments.
    """
    assert audio_times.shape[0] == len(audio_seg_ids), f"audio_times and audio_seg_ids must have the same length, but got {audio_times.shape[0]} and {len(audio_seg_ids)}."
    assert len(visual_times_vad) == len(vlabels_vad), f"visual_times_vad and vlabels_vad must have the same length, but got {len(visual_times_vad)} and {len(vlabels_vad)}."
    assert audio_times.shape[0] == S_hat_onehot.shape[0], f"audio_times and S_hat_onehot must have the same number of rows, but got {audio_times.shape[0]} and {S_hat_onehot.shape[0]}."
    # Step 1: Build dict with keys from audio_seg_ids, values as empty lists
    seg_dict = {seg_id: [] for seg_id in audio_seg_ids}

    # Step 2: For each visual segment, find which audio segment interval it falls into
    for v_idx, v_time in enumerate(visual_times_vad):
        # Find audio segment whose interval contains v_time
        match_mask = (v_time >= audio_times[:, 0]) & (v_time < audio_times[:, 1])
        if np.sum(match_mask) == 1:
            a_idx = np.where(match_mask)[0][0]
            seg_id = audio_seg_ids[a_idx]
            seg_dict[seg_id].append(vlabels_vad[v_idx])

    # Step 3: Remove entries with empty value lists and create a unique mapping
    seg_dict_vad = {seg_id: vlabel_list for seg_id, vlabel_list in seg_dict.items() if len(vlabel_list) > 0}
    seg_dict_uniq = {seg_id: int(vlabel_list[0]) for seg_id, vlabel_list in seg_dict_vad.items() if len(set(vlabel_list)) == 1}

    # Step 4: Create visual one-hot encoding matrix
    X_onehot = np.zeros_like(S_hat_onehot, dtype=int)  # same shape as S_hat_onehot
    for idx, seg_id in enumerate(audio_seg_ids):
        if seg_id in seg_dict_uniq:
            if seg_dict_uniq[seg_id] == -1:
                X_onehot[idx, -1] = 1
            else:
                X_onehot[idx, seg_dict_uniq[seg_id]] = 1

    return X_onehot

def convert_vlabels_mf_to_binary(audio_seg_ids, audio_seg_ids_mf, vlabels_mf, S_hat_onehot):
    """
    Converts mid-frame visual labels into a binary matrix mapped to audio segments.

    Args:
      audio_seg_ids (np.ndarray): A 1D array of shape (N,) containing unique identifiers for each audio segment.
      audio_seg_ids_mf (np.ndarray): A 1D array of shape (M,) containing audio segment IDs corresponding to each mid-frame faces.
      vlabels_mf (np.ndarray): A 1D array of shape (M,) containing the visual labels for each mid-frame faces. Labels are integers starting from -1 or 0, where -1 indicates 'others'.
      S_hat_onehot (np.ndarray): A one-hot encoded matrix for audio labels, used to determine the output shape.

    Returns:
      np.ndarray: A binary matrix F_hat of the same shape as S_hat_onehot. Each row corresponds to an audio segment, and columns represent mid-frame labels. A value of 1 indicates that the audio segment contains the corresponding mid-frame label.
    """
    assert len(audio_seg_ids_mf) == len(vlabels_mf), f"audio_seg_ids_mf and vlabels_mf must have the same length, but got {len(audio_seg_ids_mf)} and {len(vlabels_mf)}."
    F_hat = np.zeros_like(S_hat_onehot, dtype=int)
    for idx, seg_id in enumerate(audio_seg_ids):
        # Find indices in audio_seg_ids_mf that match the current audio segment ID
        mf_indices = np.where(audio_seg_ids_mf == seg_id)[0]
        if len(mf_indices) == 0:
            continue
        # 获取这些mid-frame的标签集合
        mf_labels = set(vlabels_mf[mf_indices])
        for label in mf_labels:
            if label == -1:
                F_hat[idx, -1] = 1
            else:
                F_hat[idx, label] = 1
    return F_hat

def convert201_together(audio_seg_ids, alabels, audio_times=None, visual_times_vad=None, vlabels_vad=None, audio_seg_ids_mf=None, vlabels_mf=None):
  """
  Converts audio and visual labels into one-hot encoded matrices for further processing.

  Args:
    audio_seg_ids (np.ndarray): A 1D array of shape (N,) containing unique identifiers for each audio segment.
    alabels (np.ndarray): A 1D array of shape (N,) containing the labels for each audio segment. Labels are integers starting from -1 or 0, where -1 indicates 'others'.  
    audio_times (np.ndarray, optional): A 2D array of shape (N, 2) where each row represents the start and end times of an audio segment.
    visual_times_vad (np.ndarray, optional): A 1D array of shape (M1,) where each element represents the timestamp of a active speaker face.
    vlabels_vad (np.ndarray, optional): A 1D array of shape (M1,) containing the labels for each active speaker face. Labels are integers starting from -1 or 0, where -1 indicates 'others'.
    audio_seg_ids_mf (np.ndarray, optional): A 1D array of shape (M2,) containing unique identifiers for audio segments that correspond to each mid-frame face.
    vlabels_mf (np.ndarray, optional): A 1D array of shape (M2,) containing the labels for each mid-frame face. Labels are integers starting from -1 or 0, where -1 indicates 'others'.

  Returns:
    tuple:
    - S_hat_onehot (np.ndarray): A 2D one-hot encoded matrix for audio labels.
    - X_onehot (np.ndarray): A 2D one-hot encoded matrix for visual labels mapped to audio segments.
    - F_hat(np.ndarray): A 2D binary matrix for mid-frame visual labels mapped to audio segments.
  """
  flag_has_neg1 = (-1 in alabels) or (vlabels_vad is not None and -1 in vlabels_vad) or (vlabels_mf is not None and -1 in vlabels_mf) # determine whether add extra column for -1 label
  n_major_clusters = np.max(alabels) + 1
  n_states = n_major_clusters + 1 if flag_has_neg1 else n_major_clusters

  # Convert labels to binary/one-hot matrices, and collect poyential sates for each time t
  # audio part
  S_hat_onehot = convert_alabels_to_onehot(audio_seg_ids, alabels, ncols=n_states)
  # visual vad part
  X_onehot = np.zeros_like(S_hat_onehot, dtype=int)
  if audio_times is not None and visual_times_vad is not None and vlabels_vad is not None:
      assert set(np.unique(vlabels_vad)).issubset(set(np.unique(alabels))), "vlabels_vad contains labels not present in alabels."
      X_onehot = convert_vlabels_vad_to_onehot(audio_times, visual_times_vad, audio_seg_ids, vlabels_vad, S_hat_onehot)
  # visual mid frame part
  F_hat = np.zeros_like(S_hat_onehot, dtype=int)
  if audio_seg_ids_mf is not None and vlabels_mf is not None:
      assert set(np.unique(vlabels_mf)).issubset(set(np.unique(alabels))), "vlabels_mf contains labels not present in alabels."
      F_hat = convert_vlabels_mf_to_binary(audio_seg_ids, audio_seg_ids_mf, vlabels_mf, S_hat_onehot)
  
  return S_hat_onehot, X_onehot, F_hat, flag_has_neg1


def collect_labels_potential(audio_seg_ids, labels_potential_list, n_states, audio_seg_ids_ref=None):
    """
    Aggregates and sorts the potential label sets for each audio segment. For potential labels with a value of -1, they are replaced with n_states-1.
    Args:
      audio_seg_ids (np.ndarray): Array of audio segment IDs with shape (N,).
      labels_potential_list (list): A list of length M, where each element is the potential label set for the corresponding sample.
      n_states (int): Total number of states, used to replace -1 labels with n_states-1.
      audio_seg_ids_ref (np.ndarray, optional): Array of audio segment IDs corresponding to the samples, with shape (M,). 
        Must be provided when the lengths of audio_seg_ids and labels_potential_list are different.
    Returns:
      list: A list of length N, where each element is a list of length n_t(number of samples corresponding to the audio segment), and each element is a sorted potential label list for that sample.
    """
    if len(audio_seg_ids) !=len(labels_potential_list):
        assert audio_seg_ids_ref is not None, "audio_seg_ids_ref cannot be None when lengths of audio_seg_ids and labels_potential_list are different."
        assert len(audio_seg_ids_ref) == len(labels_potential_list), f"audio seg_ids_mf and labels_potential_list must have the same length, but got {len(audio_seg_ids_ref)} and {len(labels_potential_list)}."
    if any([-1 in label_list for label_list in labels_potential_list]):
        assert n_states is not None, "n_states must be provided when -1 exists in labels_potential_list."
    if audio_seg_ids_ref is None:
        audio_seg_ids_ref = audio_seg_ids
    
    result = []
    for seg_id in audio_seg_ids:
        indices = np.where(audio_seg_ids_ref == seg_id)[0]
        indices = sorted(indices) # ensure the order is consistent with the original order
        label_sets = []
        for idx in indices:
            label_set = set([label if label != -1 else n_states - 1 for label in labels_potential_list[idx]])
            label_sets.append(sorted(list(label_set)))
        result.append(label_sets)
    return result

def collect_potential_states(audio_seg_ids, n_states, alabels_potential_list=None, audio_seg_ids_mf=None, vlabels_mf_potential_list=None):
  """
  Collect potential states for audio and visual mid-frame segments,and replace -1 labels with n_states-1.

  Args:
    audio_seg_ids (list): A list of audio segment IDs, length N.
    n_states (int): The number of observed/hidden states to consider in the HMM.
    alabels_potential_list (list, optional): A list of potential labels for audio segments, length N.
    audio_seg_ids_mf (list, optional): A list of audio segment IDs for mid-frame processing, length M.
    vlabels_mf_potential_list (list, optional): A list of potential labels for visual mid-frame segments, length M.

  Returns:
    tuple: A tuple containing two lists of the same length as audio_seg_ids:
      - alabels_potential_list_new (list): A list of processed potential labels for audio segments, each element is a list of labels.
      - vlabels_mf_potential_list_new (list): A list of processed potential labels for visual mid-frame segments, each element is a list of labels.
  """
  # Collect potential sates for each time t
  # audio part
  ## Only replace -1 with n_states-1 in fact
  alabels_potential_list_new = []
  if alabels_potential_list is not None:
      assert len(audio_seg_ids) == len(alabels_potential_list), f"audio_seg_ids and alabels_potential_list must have the same length, but got {len(audio_seg_ids)} and {len(alabels_potential_list)}."
      alabels_potential_list_new = collect_labels_potential(audio_seg_ids, alabels_potential_list, n_states)
  # visual mid frame part
  vlabels_mf_potential_list_new = []
  if audio_seg_ids_mf is not None and vlabels_mf_potential_list is not None:
      assert len(audio_seg_ids_mf) == len(vlabels_mf_potential_list), f"audio_seg_ids_mf and vlabels_mf_potential_list must have the same length, but got {len(audio_seg_ids_mf)} and {len(vlabels_mf_potential_list)}."
      vlabels_mf_potential_list_new = collect_labels_potential(audio_seg_ids, vlabels_mf_potential_list, n_states, audio_seg_ids_mf)
  return alabels_potential_list_new, vlabels_mf_potential_list_new

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

def save_cluster_results_vision_vad(audio_times, visual_times, audio_seg_ids, vlabels, out_json):
    """
    Save active speaker clustering results (vision_vad) to a JSON file.

    Args:
        audio_times (ndarray): [N, 2] array, each row is [start, end] for audio segments.
        visual_times (ndarray): [M,] array, each value is the time for a visual segment.
        vlabels (ndarray): [M,] array, visual cluster labels.
        audio_seg_ids (ndarray): [N,] array, segment IDs for audio segments.
        out_json (str): Path to output JSON file.
    """
    # Step 1: Build dict with keys from audio_seg_ids, values as empty lists
    seg_dict = {seg_id: [] for seg_id in audio_seg_ids}

    # Step 2: For each visual segment, find which audio segment interval it falls into
    for v_idx, v_time in enumerate(visual_times):
        # Find audio segment whose interval contains v_time
        match_mask = (v_time >= audio_times[:, 0]) & (v_time < audio_times[:, 1])
        if np.sum(match_mask) == 1:
            a_idx = np.where(match_mask)[0][0]
            seg_id = audio_seg_ids[a_idx]
            seg_dict[seg_id].append(vlabels[v_idx])

    # Step 3: Remove entries with empty value lists and create two new dicts
    seg_dict_vad = {seg_id: vlabel_list for seg_id, vlabel_list in seg_dict.items() if len(vlabel_list) > 0}
    ## Create seg_dict_uniq with value lists of length 1
    seg_dict_uniq = {seg_id: int(vlabel_list[0]) for seg_id, vlabel_list in seg_dict_vad.items() if len(set(vlabel_list)) == 1}
    ## Create seg_dict_major with the most frequent element in value lists
    seg_dict_major = {}
    for seg_id, vlabel_list in seg_dict_vad.items():
        label_counts = {label: vlabel_list.count(label) for label in set(vlabel_list)}
        major_label = max(label_counts, key=label_counts.get)
        seg_dict_major[seg_id] = int(major_label)

    ## Print comparison information
    seg_dict_len = len(seg_dict)
    seg_dict_vad_len = len(seg_dict_vad)
    seg_dict_uniq_len = len(seg_dict_uniq)
    seg_dict_major_len = len(seg_dict_major)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {current_time} Among {seg_dict_len} audio segments, {seg_dict_vad_len} segments have active visual speaker cluster labels.")
    print(f"[INFO] {current_time} Among {seg_dict_len} audio segments, {seg_dict_major_len} segments have a majority active visual speaker cluster label.")    
    print(f"[INFO] {current_time} Among {seg_dict_len} audio segments, {seg_dict_uniq_len} segments have unique active visual speaker cluster labels.")

    # Step 7: Save seg_dict_uniq and seg_dict_major to JSON
    with open(out_json.replace('.json', '_uniq.json'), 'w') as f:
        json.dump(seg_dict_uniq, f, indent=2)
    with open(out_json.replace('.json', '_major.json'), 'w') as f:
        json.dump(seg_dict_major, f, indent=2)

def save_cluster_results_vision_mf(labels, audio_seg_ids_mf, face_idxs_mf, out_json):
    """
    Save mid-frame visual clustering results to a JSON file.

    Args:
        labels (ndarray): Cluster labels for each mid-frame embedding, of shape [N].
        audio_seg_ids_mf (ndarray): Audio segment IDs corresponding to each mid-frame embedding, of shape [N].
        face_idxs_mf (ndarray): Face indices corresponding to each mid-frame embedding, of shape [N].
        out_json (str): Path to the output JSON file.

    Output:
        Saves a JSON file where each key is a combination of the audio segment ID and face index 
        (formatted as "<audio_seg_id>_<face_idx>"), and the value is the corresponding cluster label.
    """
    # Create a dictionary mapping segment ID and face index to cluster labels
    cluster_results = {f"{seg_id}_{int(face_id)}": int(label) for seg_id, face_id, label in zip(audio_seg_ids_mf, face_idxs_mf, labels)}

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
    cluster = build('cluster', config)
    labels = cluster(embeddings)
    labels = reset_cluster_ids(labels)
    summary_cluster_results(labels, modal_type='audio')
    out_json = os.path.join(result_dir, f'cluster_results_audio.json')
    save_cluster_results_audio(labels, audio_seg_ids, out_json)


def audio_vision_func_vad(local_wav_list, audio_embs_dir, visual_embs_dir, result_dir, config):
    # NOTE: length of audio_embeddings and visual_embeddings may be different
    audio_embeddings = np.array([], dtype=np.float32)
    visual_embeddings = np.array([], dtype=np.float32)
    audio_times = np.array([], dtype=np.float32)
    visual_times = np.array([], dtype=np.float32)
    audio_seg_ids = np.array([], dtype='<U50')
    alengths = []  # list of int, number of audio segments for each audio file
    # visual_infos = []   # list of tuple (rec_id, time shift, number of visual segments)

    # 对每一个音频文件，加载其对应的音频和视觉speaker embeddings，然后进行多模态聚类
    for file_idx, wav_file in enumerate(local_wav_list):
        wav_name = os.path.basename(wav_file)
        rec_id = wav_name.rsplit('.', 1)[0]
        audio_embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')
        visual_embs_file = os.path.join(visual_embs_dir, rec_id + '_vad.pkl')
        if not os.path.exists(audio_embs_file) or not os.path.exists(visual_embs_file):
            print("[WARNING]: %s or %s does not exist, it is possible that vad model did not detect valid speech or face in file %s, please check it."%(audio_embs_file, visual_embs_file, wav_file))
            continue
        
        time_begin_crt = 0 if file_idx == 0 else np.max(audio_times) + 120
        ## load embeddings
        with open(audio_embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            alengths.append(len(stat_obj['subseg_ids']))
            if file_idx == 0:
                audio_embeddings = stat_obj['embeddings']
                audio_times = stat_obj['times']
                audio_seg_ids = stat_obj['subseg_ids']
            else:
                audio_embeddings = np.vstack((audio_embeddings, stat_obj['embeddings']))
                audio_times = np.vstack((audio_times, stat_obj['times']+time_begin_crt))
                audio_seg_ids = np.hstack((audio_seg_ids, stat_obj['subseg_ids']))

        with open(visual_embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            # visual_infos.append((rec_id, time_begin_crt, len(stat_obj['embeddings'])))
            if file_idx == 0:
                visual_embeddings = stat_obj['embeddings']
                visual_times = stat_obj['times']
            else:
                visual_embeddings = np.vstack((visual_embeddings, stat_obj['embeddings']))
                visual_times = np.hstack((visual_times, stat_obj['times']+time_begin_crt))

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {current_time} For visual embeddings in {visual_embs_dir}, there are totally {len(visual_embeddings)} active face embeddings.")
    cluster = build('cluster', config)
    # visual-only clustering
    ## 聚类流程
    ## 1. 通过AHCluster(根据余弦相似度定义距离，根据提前定义的距离阈值，做层次聚类)
    ## 2. 将极小簇就近合并到较大簇
    ## 3. 根据聚类中心余弦相似度合并相似簇
    vlabels = cluster.vision_cluster(visual_embeddings)
    vlabels = reset_cluster_ids(vlabels)
    summary_cluster_results(vlabels, modal_type='visual_vad')
    save_cluster_results_vision_vad(audio_times, visual_times, audio_seg_ids, vlabels, os.path.join(result_dir, f'cluster_results_vision_vad.json'))
    
    # audio-only clustering
    ## 仍使用谱聚类实现，聚类整体流程与audio_only_func中的描述相同。min_cluster_size和pval与只有语音模态时有所不同    
    alabels = cluster.audio_cluster(audio_embeddings)
    alabels = reset_cluster_ids(alabels)
    summary_cluster_results(alabels, modal_type='audio')
    save_cluster_results_audio(alabels, audio_seg_ids, os.path.join(result_dir, f'cluster_results_audio_vision_vad_alabels.json'))

    # modify audio clustering results with visual information
    ## 具体流程
    ## 1. 计算每个audio segment与每个visual segment的overlap
    ## 2. 设置visual簇从max_audio_spk_id开始编号，筛选出至少一个visual segment与某audio簇的重叠时长>1s 的visual簇（以及与其overlap的audio segment embedding 的均值作为聚类中心）
    ## 3. 对于各个 audio 簇，查找与其重叠时长>0.5s的 visual 簇，并计算前者中各个样本与后者中各个聚类中心的余弦相似度，据此将所有audio segment分配到与其最相似的visual簇上（如果没有任何visual簇与其重叠>0.5s，则保持其audio-only聚类结果不变）。由于>0.5s的阈值并不苛刻，因此相当于利用visual信息重新分配了大部分audio segment的簇ID      
    labels, vlabels_arrange_dic = cluster(audio_embeddings, visual_embeddings, audio_times, visual_times, config, alabels, vlabels)
    del audio_embeddings, visual_embeddings
    labels_save = reset_cluster_ids(copy.deepcopy(labels))
    summary_cluster_results(labels_save, modal_type='audio_vision_vad')
    save_cluster_results_audio(labels_save, audio_seg_ids, os.path.join(result_dir, f'cluster_results_audio_vision_vad.json'))

    # Apply HMM_X smoothing to the labels
    ## 筛选出视觉簇与语音簇有所对应的视觉时间段和簇id
    vlabels_dic_aligned = {k: v for k, v in vlabels_arrange_dic.items() if v in np.unique(labels)}  # key: old id in vlabels, value: new id aligned with labels(new alabels)
    ## 提取与alabels存在对应的视觉簇id及其出现时间
    vlabels_aligned, visual_times_aligned, _ = extract_aligned_vlabels_results(vlabels, vlabels_dic_aligned, visual_times)
    ## 仅保留潜在主要说话人簇（top-2*main_actors_num），从大到小依次标记为0,1,...，其他簇统一标记为-1。将视觉簇相应重命名
    labels_processed, vlabels_processed, _ = process_top_cluster_ids_together(copy.deepcopy(labels), vlabels_aligned, vlabels_mf_aligned = None, main_actors_num = config.main_actors_num)
    S_hat_onehot, X_onehot, _, _ = convert201_together(audio_seg_ids, labels_processed, audio_times, visual_times_aligned, vlabels_processed)
    alabels_hmmX_smooth(S_hat_onehot, X_onehot, alengths, audio_seg_ids, result_dir, flag_has_neg1=(-1 in labels_processed),
                        selective_change=config.selective_change, duration_dat=audio_times[:,1] - audio_times[:,0])


def audio_vision_func_vad_mf(local_wav_list, audio_embs_dir, visual_embs_dir, result_dir, config):
    # NOTE: length of audio_embeddings and visual_embeddings_vad, visual_embeddings_mf may be different
    audio_embeddings = np.array([], dtype=np.float32)
    visual_embeddings_vad = np.array([], dtype=np.float32)
    visual_embeddings_mf = np.array([], dtype=np.float32)
    audio_times = np.array([], dtype=np.float32)
    visual_times_vad = np.array([], dtype=np.float32)
    audio_seg_ids = np.array([], dtype='<U50')  # of the same length as audio_embeddings
    audio_seg_ids_mf = np.array([], dtype='<U50')  # of the same length as visual_embeddings_mf
    face_idxs_mf = np.array([], dtype=np.int32)  # of the same length as visual_embeddings_mf
    alengths = []  # list of int, number of audio segments for each audio file
    # visual_infos = []   # list of tuple (rec_id, time shift, number of visual segments)

    # 对每一个音频文件，加载其对应的音频和视觉speaker embeddings，然后进行多模态聚类
    for file_idx, wav_file in enumerate(local_wav_list):
        wav_name = os.path.basename(wav_file)
        rec_id = wav_name.rsplit('.', 1)[0]
        audio_embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')
        visual_embs_file_vad = os.path.join(visual_embs_dir, rec_id + '_vad.pkl')
        visual_embs_file_mf = os.path.join(visual_embs_dir, rec_id + '_midframe.pkl')
        if not os.path.exists(audio_embs_file) or not os.path.exists(visual_embs_file_vad) or not os.path.exists(visual_embs_file_mf):
            print("[WARNING]: %s or %s or %sdoes not exist, it is possible that vad model did not detect valid speech or face in file %s, please check it."%(audio_embs_file, visual_embs_file_vad, visual_embs_file_mf, wav_file))
            continue
        
        time_begin_crt = 0 if file_idx == 0 else np.max(audio_times) + 120
        ## load embeddings
        with open(audio_embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            alengths.append(len(stat_obj['subseg_ids']))
            if file_idx == 0:
                audio_embeddings = stat_obj['embeddings']
                audio_times = stat_obj['times']
                audio_seg_ids = stat_obj['subseg_ids']
            else:
                audio_embeddings = np.vstack((audio_embeddings, stat_obj['embeddings']))
                audio_times = np.vstack((audio_times, stat_obj['times']+time_begin_crt))
                audio_seg_ids = np.hstack((audio_seg_ids, stat_obj['subseg_ids']))

        with open(visual_embs_file_vad, 'rb') as f:
            stat_obj = pickle.load(f)
            # visual_infos.append((rec_id, time_begin_crt, len(stat_obj['embeddings'])))
            if file_idx == 0:
                visual_embeddings_vad = stat_obj['embeddings']
                visual_times_vad = stat_obj['times']
            else:
                visual_embeddings_vad = np.vstack((visual_embeddings_vad, stat_obj['embeddings']))
                visual_times_vad = np.hstack((visual_times_vad, stat_obj['times']+time_begin_crt))

        with open(visual_embs_file_mf, 'rb') as f:
            stat_obj = pickle.load(f)
            if file_idx == 0:
                visual_embeddings_mf = stat_obj['feat']
                audio_seg_ids_mf = stat_obj['audio_seg_id'] # np.ndarray, (N, )
                face_idxs_mf = stat_obj['face_idx'] # np.ndarray, (N, )
            else:
                visual_embeddings_mf = np.vstack((visual_embeddings_mf, stat_obj['feat']))
                audio_seg_ids_mf = np.hstack((audio_seg_ids_mf, stat_obj['audio_seg_id']))
                face_idxs_mf = np.hstack((face_idxs_mf, stat_obj['face_idx']))


    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {current_time} For visual embeddings in {visual_embs_dir}, there are totally {len(visual_embeddings_vad)} active face embeddings, and {len(visual_embeddings_mf)} mid-frame face embeddings.")

    # create cluster object for audio-visual(vad) joint clustering and visual(mid-frame) clustering
    config_mf = copy.deepcopy(config)
    config_mf.vision_cluster['args']['fix_cos_thr'] = config_mf.fix_cos_thr_mf
    del config_mf.audio_cluster, config_mf.cluster

    cluster = build('cluster', config)
    cluster_mf = build('vision_cluster', config_mf)

    # visual-only clustering
    ## 聚类流程
    ## 1. 通过AHCluster(根据余弦相似度定义距离，根据提前定义的距离阈值，做层次聚类)
    ## 2. 将极小簇就近合并到较大簇
    ## 3. 根据聚类中心余弦相似度合并相似簇

    ## active speaker face clustering
    vlabels_vad = cluster.vision_cluster(visual_embeddings_vad)
    vlabels_vad = reset_cluster_ids(vlabels_vad)
    summary_cluster_results(vlabels_vad, modal_type='visual_vad')
    save_cluster_results_vision_vad(audio_times, visual_times_vad, audio_seg_ids, vlabels_vad, os.path.join(result_dir, f'cluster_results_vision_vad.json'))
    ## mid-frame face clustering
    vlabels_mf = cluster_mf(visual_embeddings_mf)
    vlabels_mf = reset_cluster_ids(vlabels_mf)
    summary_cluster_results(vlabels_mf, modal_type='visual_mid_frame_before_vision_align')
    save_cluster_results_vision_mf(vlabels_mf, audio_seg_ids_mf, face_idxs_mf, os.path.join(result_dir, f'cluster_results_faces_mid_frame_before_vision_align.json'))

    ## align mid-frame face clustering results with active speaker face clustering results
    ### 根据两种视觉聚类结果的聚类中心余弦相似度进行对齐
    align_cos_thr=0.5
    print(f"[INFO] Set cos-similarity threshold  to {align_cos_thr} during aligning mid-frame faces clustering and active speaker face clustering.")
    vlabels_mf = align_clusters2clusters(copy.deepcopy(vlabels_mf), copy.deepcopy(vlabels_vad), visual_embeddings_mf, visual_embeddings_vad, align_cos_thr=align_cos_thr)
    summary_cluster_results(vlabels_mf, modal_type='visual_mid_frame_vision_aligned')
    save_cluster_results_vision_mf(vlabels_mf, audio_seg_ids_mf, face_idxs_mf, os.path.join(result_dir, f'cluster_results_faces_mid_frame_vision_aligned.json'))
    

    # audio-only clustering
    ## 仍使用谱聚类实现，聚类整体流程与audio_only_func中的描述相同。min_cluster_size和pval与只有语音模态时有所不同    
    alabels = cluster.audio_cluster(audio_embeddings)
    alabels = reset_cluster_ids(alabels)
    summary_cluster_results(alabels, modal_type='audio')
    save_cluster_results_audio(alabels, audio_seg_ids, os.path.join(result_dir, f'cluster_results_audio_vision_vad_alabels.json'))

    # modify audio clustering results with visual information
    ## 具体流程
    ## 1. 计算每个audio segment与每个visual segment的overlap
    ## 2. 设置visual簇从max_audio_spk_id开始编号，筛选出至少一个visual segment与某audio簇的重叠时长>1s 的visual簇（以及与其overlap的audio segment embedding 的均值作为聚类中心）
    ## 3. 对于各个 audio 簇，查找与其重叠时长>0.5s的 visual 簇，并计算前者中各个样本与后者中各个聚类中心的余弦相似度，据此将所有audio segment分配到与其最相似的visual簇上（如果没有任何visual簇与其重叠>0.5s，则保持其audio-only聚类结果不变）。由于>0.5s的阈值并不苛刻，因此相当于利用visual信息重新分配了大部分audio segment的簇ID
    alabels, vlabels_vad_arrange_dic = cluster(audio_embeddings, visual_embeddings_vad, audio_times, visual_times_vad, config, alabels, vlabels_vad)
    del visual_embeddings_vad
    alabels_save = reset_cluster_ids(copy.deepcopy(alabels))
    summary_cluster_results(alabels_save, modal_type='audio_vision_vad')
    save_cluster_results_audio(alabels_save, audio_seg_ids, os.path.join(result_dir, f'cluster_results_audio_vision_vad.json'))


    # Apply HMM_nested_X smoothing to the alabels and vlabels_mf
    ## 将 mid-frame 视觉簇、 active speaker face 视觉簇与语音簇的id对齐
    ### 从 active speaker face 视觉簇中，筛选与语音簇有所对应的的部分，获取其簇id以及其中每一个样本的视觉出现时间
    vlabels_vad_dic_aligned = {k: v for k, v in vlabels_vad_arrange_dic.items() if v in np.unique(alabels)}  # key: old id in vlabels, value: new id aligned with labels(new alabels)
    vlabels_vad_aligned, visual_times_vad_aligned, _ = extract_aligned_vlabels_results(vlabels_vad, vlabels_vad_dic_aligned, visual_times_vad)
    summary_cluster_results(vlabels_vad_aligned, modal_type='visual_vad_vision-audio_aligned')
    save_cluster_results_vision_vad(audio_times, visual_times_vad_aligned, audio_seg_ids, vlabels_vad_aligned, 
                                    os.path.join(result_dir, f'cluster_results_vision_vad_vision-audio_aligned.json'))
    ### 由于前面已经对齐了 mid-frame 视觉簇与 active speaker face 视觉簇，因此可以直接利用 active speaker face 视觉簇与语音簇的对应关系，来对齐 mid-frame 视觉簇与语音簇
    vlabels_mf_aligned, _, aligned_mask_mf = extract_aligned_vlabels_results(vlabels_mf, vlabels_vad_dic_aligned, None) # only keep aligned mf labels, which is indicated by aligned_mask_mf
    audio_seg_ids_mf_aligned, face_idxs_mf_aligned = audio_seg_ids_mf[aligned_mask_mf], face_idxs_mf[aligned_mask_mf]
    summary_cluster_results(vlabels_mf_aligned, modal_type='visual_mid_frame_vision-audio_aligned')
    save_cluster_results_vision_mf(vlabels_mf_aligned, audio_seg_ids_mf_aligned, face_idxs_mf_aligned, 
                                   os.path.join(result_dir, f'cluster_results_faces_mid_frame_vision-audio_aligned.json'))

    ## 仅保留潜在主要说话人簇（top-2*main_actors_num），从大到小依次标记为0,1,...，其他簇统一标记为-1，最终得到2*main_actors_num+1个类。将视觉簇相应重命名
    alabels_processed, vlabels_vad_processed, vlabels_mf_processed = process_top_cluster_ids_together(copy.deepcopy(alabels), vlabels_vad_aligned, vlabels_mf_aligned, main_actors_num = config.main_actors_num)
    summary_cluster_results(alabels_processed, modal_type='audio_processed_for_HMM_nested_X')
    summary_cluster_results(vlabels_vad_processed, modal_type='visual_vad_processed_for_HMM_nested_X')
    summary_cluster_results(vlabels_mf_processed, modal_type='visual_mid_frame_processed_for_HMM_nested_X')
    # save_cluster_results_audio(alabels_processed, audio_seg_ids, os.path.join(result_dir, f'cluster_results_audio_processed_for_HMM_nested_X.json'))
    # save_cluster_results_vision_vad(audio_times, visual_times_vad_aligned, audio_seg_ids, vlabels_vad_processed, 
    #                                os.path.join(result_dir, f'cluster_results_vision_vad_processed_for_HMM_nested_X.json'))
    # save_cluster_results_vision_mf(vlabels_mf_processed, audio_seg_ids_mf_aligned, face_idxs_mf_aligned, 
    #                                os.path.join(result_dir, f'cluster_results_faces_mid_frame_processed_for_HMM_nested_X.json'))

    ## 获取audio samples cluster result的unreliable metrics
    alabels_unreliable_metrics = get_unreliable_metrics(copy.deepcopy(alabels_processed), audio_embeddings)
    ## 获取所有audio samples和前述对齐和重命名处理后，剩余的所有关键帧人脸 samples，并据此创建每个sample潜在对应的聚类簇候选集
    ### 所得候选集中的cluster id除了-1之外，与后面HMM states的state id一一对应。-1对应HMM states中的n_states-1
    ### NOTE: 如果改用一般的align_samples2clusters，将target cluster设置为avd，则需要check后面对于-1的处理
    alabels_potential_list = align_samples2clusters(copy.deepcopy(alabels_processed), audio_embeddings,
                                                        candi_align_cluster_num=3) # of the same length as alabels_processed
    vlabels_mf_potential_list = align_samples2clusters(copy.deepcopy(vlabels_mf_processed), visual_embeddings_mf[aligned_mask_mf],
                                                            candi_align_cluster_num=3) # of the same length as vlabels_mf_processed
    del audio_embeddings, visual_embeddings_mf

    ## 转换观测及avd协变量为 binary 编码矩阵
    S_hat_onehot, X_onehot, F_hat, flag_has_neg1 = convert201_together(audio_seg_ids, alabels_processed, 
                                                                       audio_times, visual_times_vad_aligned, vlabels_vad_processed, 
                                                                       audio_seg_ids_mf_aligned, vlabels_mf_processed)
    ## 收集每个audio segment中 S_t, F_t 的潜在状态集合，并将-1替换为n_states-1
    S_potential_list, F_potential_list = collect_potential_states(audio_seg_ids, S_hat_onehot.shape[1], alabels_potential_list, 
                                                                  audio_seg_ids_mf_aligned, vlabels_mf_potential_list)
    
    # np.save(os.path.join(result_dir, 'cluster_results_face_states_obs.npy'), F_hat)
    ## HMM_nested_X 平滑
    
    for params in ["cehdfij"]:
        labels_nested_hmm_full_smooth(S_hat_onehot, F_hat, X_onehot, alengths, params, audio_seg_ids, result_dir, flag_has_neg1=flag_has_neg1, alabels_unreliable_metrics=alabels_unreliable_metrics)
        labels_nested_hmm_full_smooth(S_hat_onehot, F_hat, X_onehot, alengths, params, audio_seg_ids, result_dir, flag_has_neg1=flag_has_neg1, alabels_unreliable_metrics=alabels_unreliable_metrics, B_S_diag_min=0.7)


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
        assert args.visual_info_type in ['vad', 'key_frame', 'vad+key_frame'], f'--visual_info_type should be either "vad", "key_frame" or "vad+key_frame", but got {args.visual_info_type}'
        if args.visual_info_type == 'vad':
            audio_vision_func_vad(wav_list, args.audio_embs_dir, args.visual_embs_dir, args.result_dir, config)
        elif args.visual_info_type == 'key_frame':
            raise NotImplementedError("Not implemented yet.")
        else:
            audio_vision_func_vad_mf(wav_list, args.audio_embs_dir, args.visual_embs_dir, args.result_dir, config)


if __name__ == "__main__":
    main()
