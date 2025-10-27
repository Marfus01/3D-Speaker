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
import json
from datetime import datetime
from speakerlab.process.hmm.hmm_X import HMM_X
from speakerlab.process.hmm.nested_hmm import NestedHMM
from speakerlab.process.hmm.nested_hmm_full import NestedHMM_full

parser = argparse.ArgumentParser(description='Cluster embeddings and output rttm files')
parser.add_argument('--conf', default=None, help='Config file')
parser.add_argument('--wavs', default=None, help='Wav list file')
parser.add_argument('--cluster_type', default='audio_only', type=str, help='Clustering type, support "audio_only" and "audio_vision"')
parser.add_argument('--visual_info_type', default='vad', type=str, help='Visual information type, support "vad", "key_frame", "vad+key_frame"')
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

class SuppressMultinomialHMMWarning:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

def alabels_hmmX_smooth(S_hat_onehot, X_onehot, lengths, audio_seg_ids, result_dir, flag_has_neg1=False,
                        prop_keep_list=[0.1], duration_dat= None):
    n_actors = S_hat_onehot.shape[1]    
    alabels = np.argmax(S_hat_onehot, axis=1)
    
    n_states_obs = n_actors
    n_states_hid = n_states_obs    
    speaker_obs = alabels.reshape(-1, 1)

    # define initial probs in hmm
    audio_startprob_init = np.ones(n_states_hid) / n_states_hid  # uniform distribution, of shape (n_stn_states_hidates,)
    audio_transitionprob_init = np.ones((n_states_hid, n_states_hid)) * (1 - 0.4) / n_states_hid
    audio_emissionprob_init = np.ones((n_states_hid, n_states_obs)) * 0.4 / n_states_obs
    # round these probs to 5 decimal places
    audio_startprob_init = np.round(audio_startprob_init, 5)
    audio_transitionprob_init = np.round(audio_transitionprob_init, 5)
    audio_emissionprob_init = np.round(audio_emissionprob_init, 5)
    # make sure the sum of each row is 1
    audio_startprob_init[0] = 1 - np.sum(audio_startprob_init[1:])
    for i in range(n_states_hid):
        audio_transitionprob_init[i, i] = 1 - np.sum(audio_transitionprob_init[i]) + audio_transitionprob_init[i, i]
        audio_emissionprob_init[i, i] = 1 - np.sum(audio_emissionprob_init[i]) + audio_emissionprob_init[i, i]    
    with SuppressMultinomialHMMWarning():
        audio_hmm_model = hmm.CategoricalHMM(n_components=n_states_hid, 
                                    n_iter=1000, tol=0.00001,
                                    init_params='')  # don't use default initial parameters

    audio_hmm_model.n_features = n_states_obs
    audio_hmm_model.startprob_ = audio_startprob_init
    audio_hmm_model.transmat_ = audio_transitionprob_init
    audio_hmm_model.emissionprob_ = audio_emissionprob_init

    audio_hmm_model.fit(speaker_obs, lengths)
    pred_speaker_hmm = audio_hmm_model.predict(speaker_obs, lengths)
    if flag_has_neg1:  # 将-1标签还原回来
        pred_speaker_hmm[pred_speaker_hmm == n_actors - 1] = -1 

    # Save the smoothed alabels to a new JSON file
    smoothed_cluster_dic = {seg_id: int(label) for seg_id, label in zip(audio_seg_ids, pred_speaker_hmm)}
    with open(os.path.join(result_dir, f'cluster_results_audio_vision_vad_hmm_naive.json'), 'w', encoding='utf-8') as f:
        json.dump(smoothed_cluster_dic, f, indent=2)

    # print("\n=== 训练模型 ===")
    # start_time = time.time()
    # print("训练开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    # model = HMM_X(n_actors=n_actors, n_iter=100, tol=1e-3, verbose=True)
    # model.fit(S_hat_onehot, X_onehot, lengths)
    # end_time = time.time()
    # print("训练结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    # print("训练耗时:", end_time - start_time, "秒")

    # print("\n=== 模型参数 ===")
    # print("说话人初始概率 β 的logits：\n", model.beta_)
    # print("说话人转移矩阵 A_S_ 的logits：\n", model.A_S_)
    # print("说话人识别混淆矩阵 B_S :\n", model.B_S_)
    # print("协变量X取值为1对说话人初始状态的影响 η1_ :\n", model.eta1_)
    # print("协变量X取值为1对说话人转移的影响 η2_ :\n", model.eta2_)

    # # 使用训练好的模型解码隐藏状态及其后验概率
    # pred_probs = model.predict_proba(S_hat_onehot, X_onehot, lengths)['speaker_states'] # 计算后验概率  (n_samples, n_states_hid)
    # speaker_states_viterbi = model.predict(S_hat_onehot, X_onehot, lengths) # viterbi 解码结果
    # speaker_states_viterbi_prob = np.array([pred_probs[i, state] for i, state in enumerate(speaker_states_viterbi)])
    # if flag_has_neg1:  # 将-1标签还原回来
    #     speaker_states_viterbi[speaker_states_viterbi == n_actors - 1] = -1 
    #     alabels[alabels == n_actors - 1] = -1

    # # Find the indices of the top prop_keep proportion of probabilities
    # num_samples = len(speaker_states_viterbi_prob)
    # for prop_keep in prop_keep_list:
    #     alabels_smoothed = copy.deepcopy(alabels)
    #     num_keep = int(num_samples * prop_keep)
    #     if num_keep > 0:
    #         # indices of the hidden states with the highest probabilities
    #         top_indices = np.argsort(speaker_states_viterbi_prob)[-num_keep:] 
    #         # select only those indices where duration_dat <= 1 second, if duration_dat is provided
    #         if duration_dat is not None:
    #             top_indices = [i for i in top_indices if duration_dat[i] <= 1]
    #         # Replace the values in the observed sequence at these indices
    #         replace_cnt = (alabels_smoothed[top_indices] != speaker_states_viterbi[top_indices]).sum()
    #         print(f"Prop keep: {float(prop_keep*100)}%, replace count: {replace_cnt}")
    #         alabels_smoothed[top_indices] = speaker_states_viterbi[top_indices]
        
    #     # Save the smoothed alabels to a new JSON file
    #     smoothed_cluster_dic = {seg_id: int(label) for seg_id, label in zip(audio_seg_ids, alabels_smoothed)}
    #     with open(os.path.join(result_dir, f'cluster_results_audio_vision_vad_hmm_prop{float(prop_keep*100)}.json'), 'w', encoding='utf-8') as f:
    #         json.dump(smoothed_cluster_dic, f, indent=2)


def extract_aligned_avd_results(visual_times, vlabels, vlabels_aligned_dic):
    """
    Filter and align visual_times and vlabels based on a mapping dictionary.

    This function filters the visual_times and vlabels arrays to include only the rows where
    the vlabels values are present as keys in the vlabels_aligned_dic. It then maps the filtered
    vlabels to new labels using the dictionary.

    Args:
        visual_times (np.array): A 1D numpy array of length n, representing visual segment times.
        vlabels (np.array): A 1D numpy array of length n, representing visual cluster labels.
        vlabels_aligned_dic (dict): A dictionary where keys are a subset of vlabels, and values
                                    are the corresponding new labels.

    Returns:
        np.array: A 1D numpy array of length m, filtered visual_times.
        np.array: A 1D numpy array of length m, mapped vlabels_new.
    """
    # Filter rows where vlabels are in vlabels_aligned_dic keys
    mask = np.isin(vlabels, list(vlabels_aligned_dic.keys()))
    filtered_visual_times = visual_times[mask]
    filtered_vlabels = vlabels[mask]

    # Map filtered vlabels to new labels using the dictionary
    vlabels_new = np.array([vlabels_aligned_dic[label] for label in filtered_vlabels])

    return filtered_visual_times, vlabels_new

def summary_cluster_results(labels, modal_type='audio'):
    """
    Summary statistics of cluster sizes
    """
    uniq = np.unique(labels)
    # Count occurrences of each unique label
    cluster_sizes = {label: np.sum(labels == label) for label in uniq}
    # Sort labels by their occurrence counts in descending order
    sorted_label_counts = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

    # print total number of clusters and total number of samples
    clusters_num = len(uniq)
    total_samples = len(labels)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {current_time} {modal_type} clustering results: total {clusters_num} clusters, total {total_samples} samples.")

    # Print the top 20 labels with their counts
    print(f"[INFO] Detailed {modal_type} cluster sizes:")
    print(f"[INFO] Top 20 label counts:")
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

def reset_cluster_ids(labels):
    """
    Reset cluster IDs to be consecutive integers starting from 0.

    Args:
        labels (ndarray): Original cluster labels, of shape [N].

    Returns:
        ndarray: New cluster labels with consecutive integers starting from 0, of shape [N].
    """
    uniq = np.unique(labels)
    # Count occurrences of each unique label
    uniq_count = {label: np.sum(labels == label) for label in uniq}
    # Sort labels by count (descending), then by label value (descending)
    sorted_uniq = sorted(uniq_count.keys(), key=lambda x: (-uniq_count[x], -x))

    new_labels = np.zeros(len(labels), dtype=int)
    for new_id, old_id in enumerate(sorted_uniq):
        new_labels[labels==old_id] = new_id
    return new_labels

def process_top_cluster_ids_together(alabels, vlabels_aligned, main_actors_num=1):
  """
  Reset cluster IDs to be consecutive integers starting from -1, while retaining only the top clusters.

  This function processes two sets of cluster labels (`alabels` and `vlabels_aligned`) and ensures that only the top 
  clusters (based on size) are retained. The remaining clusters are assigned a label of -1. The function also ensures 
  that all labels in `vlabels_aligned` are a subset of those in `alabels`.

    alabels (ndarray): Array of original cluster labels for audio, of shape [N_a].
    vlabels_aligned (ndarray): Array of aligned cluster labels for video, of shape [N_v].
    main_actors_num (int, optional): The number of main actors to retain. Only the top `2 * main_actors_num` clusters 
                      (based on size) will be retained. Defaults to 1.

    tuple: A tuple containing:
      - new_alabels (ndarray): Updated cluster labels for audio, with non-top clusters set to -1.
      - new_vlabels_aligned (ndarray): Updated cluster labels for video, with non-top clusters set to -1.

  Raises:
    AssertionError: If `vlabels_aligned` contains labels not present in `alabels`.
  """
  uniq_a = np.unique(alabels)
  uniq_v = np.unique(vlabels_aligned)
  assert set(uniq_v).issubset(set(uniq_a)), "vlabels_aligned contains labels not present in alabels."

  # Count occurrences of each unique alabel
  uniq_a_count = {aid: np.sum(alabels == aid) for aid in uniq_a}
  # Sort alabels by count (descending), then by aid value (descending)
  sorted_uniq_a = sorted(uniq_a_count.keys(), key=lambda x: (-uniq_a_count[x], -x))

  new_alabels = np.full(len(alabels), -1, dtype=int)  # Default all alabels to -1
  new_vlabels_aligned = np.full(len(vlabels_aligned), -1, dtype=int)  # Default all vlabels_aligned to -1
  # Retain only the top 2 * main_actors_num clusters
  top_clusters = sorted_uniq_a[:2 * main_actors_num]
  for new_id, old_id in enumerate(top_clusters):
    new_alabels[alabels == old_id] = new_id
    new_vlabels_aligned[vlabels_aligned == old_id] = new_id

  return new_alabels, new_vlabels_aligned

def convert201_together(audio_times, visual_times, audio_seg_ids, alabels, vlabels):
    """
    Converts audio and visual labels into one-hot encoded matrices for further processing.
    This function processes audio and visual data to create two one-hot encoded matrices:
    - `S_hat_onehot`: Represents the one-hot encoding of audio labels (`alabels`) for each audio segment.
    - `X_onehot`: Represents the one-hot encoding of visual labels (`vlabels`) mapped to corresponding audio segments.
    Args:
      audio_times (np.ndarray): A 2D array of shape (N, 2) where each row represents the start and end times 
                    of an audio segment.
      visual_times (np.ndarray): A 1D array of shape (M,) where each element represents the timestamp of a 
                    visual segment.
      audio_seg_ids (list): A list of unique identifiers for each audio segment.
      alabels (np.ndarray): A 1D array of shape (N,) containing the labels for each audio segment. Labels 
                  are integers starting from -1, where -1 indicates no label.
      vlabels (np.ndarray): A 1D array of shape (M,) containing the labels for each visual segment. Labels 
                  are integers starting from -1, where -1 indicates no label.
    Returns:
      tuple:
        - S_hat_onehot (np.ndarray): A 2D one-hot encoded matrix of shape (N, K+2), where K is the maximum 
                      label in `alabels`. The last column corresponds to the -1 label.
        - X_onehot (np.ndarray): A 2D one-hot encoded matrix of shape (N, K+2), where K is the maximum 
                    label in `alabels`. The last column corresponds to the -1 label. This matrix 
                    maps visual labels to their corresponding audio segments.
    Raises:
      AssertionError: If `alabels` does not contain consecutive integers starting from -1.
      AssertionError: If `vlabels` contains labels not present in `alabels`.
    Notes:
      - The function assumes that `audio_times` and `audio_seg_ids` are aligned, i.e., the i-th row of 
        `audio_times` corresponds to the i-th element of `audio_seg_ids`.
      - Visual segments (`visual_times`) are mapped to audio segments based on their timestamps.
      - If multiple visual labels map to the same audio segment, the segment is ignored unless all visual 
        labels are identical.
    """
    n_major_clusters = np.max(alabels) + 1
    if -1 in alabels:
        assert set(np.unique(alabels)) == set(range(-1, n_major_clusters)), "alabels contains non-consecutive integers starting from -1."
    else:
        assert set(np.unique(alabels)) == set(range(0, n_major_clusters)), "alabels contains non-consecutive integers starting from 0."
    assert set(np.unique(vlabels)).issubset(set(np.unique(alabels))), "vlabels contains labels not present in alabels."
    ####### audio parts #######
    if -1 in alabels:
        S_hat_onehot = np.zeros((len(audio_seg_ids), n_major_clusters + 1), dtype=int)  # last column for -1 label, each row corresponds to a segment
    else:
        S_hat_onehot = np.zeros((len(audio_seg_ids), n_major_clusters), dtype=int)  # each row corresponds to a segment

    for idx, label in enumerate(alabels):
        if label == -1:
            S_hat_onehot[idx, -1] = 1
        else:
            S_hat_onehot[idx, label] = 1

    ####### visual parts #######
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

    # step 4: Create visual one-hot encoding matrix
    X_onehot = np.zeros_like(S_hat_onehot, dtype=int)  # same shape as S_hat_onehot
    for idx, seg_id in enumerate(audio_seg_ids):
        if seg_id in seg_dict_uniq:
            if seg_dict_uniq[seg_id] == -1:
                X_onehot[idx, -1] = 1
            else:
                X_onehot[idx, seg_dict_uniq[seg_id]] = 1
            
    return S_hat_onehot, X_onehot

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
    del visual_embeddings
    visual_embeddings = None # 释放内存
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
    del audio_embeddings
    labels_save = reset_cluster_ids(copy.deepcopy(labels))
    summary_cluster_results(labels_save, modal_type='audio_vision_vad')
    save_cluster_results_audio(labels_save, audio_seg_ids, os.path.join(result_dir, f'cluster_results_audio_vision_vad.json'))

    # Apply HMM_X smoothing to the labels
    ## 筛选出视觉簇与语音簇有所对应的视觉时间段和簇id
    vlabels_dic_aligned = {k: v for k, v in vlabels_arrange_dic.items() if v in np.unique(labels)}  # key: old id in vlabels, value: new id aligned with labels(new alabels)
    ## 提取与alabels存在对应的视觉簇id及其出现时间
    visual_times_aligned, vlabels_aligned = extract_aligned_avd_results(visual_times, vlabels, vlabels_dic_aligned)
    summary_cluster_results(vlabels_aligned, modal_type='vlabels_aligned')
    save_cluster_results_vision_vad(audio_times, visual_times_aligned, audio_seg_ids, vlabels_aligned, os.path.join(result_dir, f'cluster_results_vision_vad_aligned.json'))
    ## 仅保留潜在主要说话人簇（top-2*main_actors_num），从大到小依次标记为0,1,...，其他簇统一标记为-1。将视觉簇相应重命名
    labels_processed, vlabels_processed = process_top_cluster_ids_together(copy.deepcopy(labels), vlabels_aligned, main_actors_num= config.main_actors_num)
    summary_cluster_results(labels_processed, modal_type='labels_processed')
    save_cluster_results_audio(labels_processed, audio_seg_ids, os.path.join(result_dir, f'cluster_results_audio_vision_vad_processed.json'))
    summary_cluster_results(vlabels_processed, modal_type='vlabels_processed')
    save_cluster_results_vision_vad(audio_times, visual_times_aligned, audio_seg_ids, vlabels_processed, os.path.join(result_dir, f'cluster_results_vision_vad_processed.json'))
    S_hat_onehot, X_onehot = convert201_together(audio_times, visual_times_aligned, audio_seg_ids, labels_processed, vlabels_processed)
    print(f"First 500 audio_seg_ids: {audio_seg_ids[:500]}")
    alabels_hmmX_smooth(S_hat_onehot, X_onehot, alengths, audio_seg_ids, result_dir, flag_has_neg1=(-1 in labels_processed),
                        prop_keep_list=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 1], duration_dat=None)

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
            raise NotImplementedError("Not implemented yet.")


if __name__ == "__main__":
    main()
