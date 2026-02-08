# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script implements baseline clustering methods for speaker diarization,
including both original versions and self-supervised learning extensions.

Baseline methods:
- CAM++ & SC (Spectral Clustering)
- CAM++ & VBx
- CAM++ & SC + k-means clustering with visual centers
- CAM++ & Pairwise Constrained Clustering (PCC)
- CAM++ & CurricularFace & joint clustering(baseline_joint)
"""

import os
import sys
import copy
import json
import pickle
import shutil
import argparse
import numpy as np
from datetime import datetime

current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.process.cluster import summary_cluster_results, reset_cluster_ids, align_clusters2clusters
from speakerlab.process.vbx.vbx_enhancer import VBxEnhancer
from speakerlab.process.cluster import ConstrainedCluster
from speakerlab.process.cluster import JointClustering_baseline

# Import utility functions from the original cluster_and_postprocess.py
from cluster_and_postprocess import (
    save_cluster_results_audio, save_cluster_results_vision_mf, save_cluster_results_vision_vad,
    load_embeds_audio, load_embed_vision_vad, load_embeds_vision_mf
)

parser = argparse.ArgumentParser(description='Baseline clustering methods for speaker diarization')
parser.add_argument('--conf', default=None, help='Config file')
parser.add_argument('--wavs', default=None, help='Wav list file')
parser.add_argument('--baseline_method', required=True, type=str, choices=['sc', 'vbx', 'kcenter', 'pcc', 'joint'], help='Baseline method to run')
parser.add_argument('--audio_embs_dir', default=None, type=str, help='Audio embedding dir')
parser.add_argument('--visual_embs_dir', default=None, type=str, help='Visual embedding dir')
parser.add_argument('--result_dir', default=None, type=str, help='Result dir')
parser.add_argument('--from_preds', action='store_true', 
                   help='Use predictions from fine-tuned model instead of re-clustering')

def load_alabels_embeddings_ft(alabels_embeddings_ft_path):
    """
    Load audio embeddings, with support for loading from previous clustering step if available.
    Args:
        - alabels_embeddings_ft_path: Path to the pickle file containing audio embeddings after fine-tuning
    Returns:
        - audio_embeddings: numpy array of shape (num_segments, embedding_dim)
        - audio_seg_ids: list of segment ids corresponding to the embeddings
        - useful_var_dic: dictionary containing useful variables copied from previous clustering step (e.g., audio_times, time_begin_crt_list)`
    """
    with open(alabels_embeddings_ft_path, 'rb') as f:
        audio_embeddings_dic = pickle.load(f)
    
    ## load useful variables copied from previous clustering step
    result_dir = os.path.dirname(alabels_embeddings_ft_path)
    useful_var_path = os.path.join(result_dir, 'useful_var_dic.pkl')
    assert os.path.exists(useful_var_path), f"When from_preds is True, useful_var_dic.pkl must exist in {result_dir}."
    with open(useful_var_path, 'rb') as f:
        useful_var_dic = pickle.load(f)
    
    audio_seg_ids = useful_var_dic['audio_seg_ids']
    audio_embeddings = np.array([audio_embeddings_dic[seg_id] for seg_id in audio_seg_ids])
    
    return audio_embeddings, audio_seg_ids, useful_var_dic

def load_alabels_preds(result_dir, audio_seg_ids):
    """
    Load audio segment predictions from fine-tuned model.
    
    Args:
        - result_dir: Directory where the predictions are stored
        - audio_seg_ids: List of audio segment ids to get predictions for
    Returns:
        - alabels_pred: numpy array of predicted labels corresponding to audio_seg_ids
    """
    alabels_pred_dic_path = os.path.join(result_dir, 'alabels_pred_dic.pkl')
    assert os.path.exists(alabels_pred_dic_path), f"alabels_pred_dic.pkl must exist in {result_dir} when loading predictions."
    with open(alabels_pred_dic_path, 'rb') as f:
        alabels_pred_dic = pickle.load(f)
    alabels_pred = np.array([alabels_pred_dic[seg_id] for seg_id in audio_seg_ids])
    return alabels_pred

def baseline_audio_sc(local_wav_list, audio_embs_dir, result_dir, config):
    """
    Baseline: CAM++ & SC (Spectral Clustering)
    
    仅使用语音embedding进行谱聚类，不做任何后处理。
    
    Args:
        local_wav_list: List of wav file paths
        audio_embs_dir: Directory containing audio embeddings
        result_dir: Directory to save results
        config: Configuration object
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {current_time} Running baseline: CAM++ & SC")
    
    # Load embeddings
    alabels_embeddings_ft_path = os.path.join(result_dir, 'alabels_embeddings.pkl')
    if os.path.exists(alabels_embeddings_ft_path):
        audio_embeddings, audio_seg_ids, useful_var_dic = load_alabels_embeddings_ft(alabels_embeddings_ft_path)
    else:
        audio_embeddings, audio_seg_ids, _, _, _ = load_embeds_audio(local_wav_list, audio_embs_dir)

    # Spectral clustering
    cluster = build('cluster', config)
    alabels = cluster(audio_embeddings)
    alabels = reset_cluster_ids(alabels)

    # Save results
    summary_cluster_results(alabels, modal_type='audio_baseline_sc')
    out_json = os.path.join(result_dir, 'pseudo_labels_audio_baseline_sc.json')
    save_cluster_results_audio(alabels, audio_seg_ids, out_json)

    # Save useful variables
    useful_var_dic = {}
    useful_var_dic['audio_seg_ids'] = audio_seg_ids
    useful_var_dic['alabels'] = alabels # can be used for get mapping between old cluster ids and new cluster ids in future, don't apply now
    useful_var_path = os.path.join(result_dir, 'useful_var_dic.pkl')
    with open(useful_var_path, 'wb') as f:
        pickle.dump(useful_var_dic, f)


def baseline_audio_vbx(local_wav_list, audio_embs_dir, result_dir, config, from_preds=False):
    """
    Baseline: CAM++ & VBx
    
    使用谱聚类初始化，然后用VBx进行平滑。
    
    Args:
        local_wav_list: List of wav file paths
        audio_embs_dir: Directory containing audio embeddings
        result_dir: Directory to save results
        config: Configuration object
        from_preds: If True, use predictions from model as initialization
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {current_time} Running baseline: CAM++ & VBx")
    
    # Load embeddings
    alabels_embeddings_ft_path = os.path.join(result_dir, 'alabels_embeddings.pkl')
    if os.path.exists(alabels_embeddings_ft_path):
        audio_embeddings, audio_seg_ids, useful_var_dic = load_alabels_embeddings_ft(alabels_embeddings_ft_path)
    else:
        audio_embeddings, audio_seg_ids, _, _, _ = load_embeds_audio(local_wav_list, audio_embs_dir)

    # Get initial labels for VBx
    suffix = '_from_model' if from_preds else ''
    if not from_preds:
        # Use spectral clustering for initialization
        cluster = build('cluster', config)
        initial_labels = cluster(audio_embeddings)
        initial_labels = reset_cluster_ids(initial_labels)
    else:
        # Use model predictions for initialization
        initial_labels = load_alabels_preds(result_dir, audio_seg_ids)
    
    summary_cluster_results(initial_labels, modal_type=f'audio_baseline_vbx_initial{suffix}')
    out_json = os.path.join(result_dir, f'cluster_results_audio_baseline_vbx_initial{suffix}.json')
    save_cluster_results_audio(initial_labels, audio_seg_ids, out_json)
    
    # Apply VBx smoothing
    vbx = VBxEnhancer(
        lda_dim=128,
        Fa=0.40,
        Fb=64,
        loopP=0.65,
        num_em_iters=10,
        init_smoothing=5.0,
        max_iters=20
    )
    labels_smoothed = vbx.fit_predict(audio_embeddings, initial_labels)
    
    # Save results
    summary_cluster_results(labels_smoothed, modal_type='audio_baseline_vbx')
    out_json = os.path.join(result_dir, f'pseudo_labels_audio_baseline_vbx{suffix}.json')
    save_cluster_results_audio(labels_smoothed, audio_seg_ids, out_json)

    # Save useful variables
    useful_var_dic = {}
    useful_var_dic['audio_seg_ids'] = audio_seg_ids
    useful_var_dic['alabels'] = labels_smoothed # can be used for get mapping between old cluster ids and new cluster ids in future, don't apply now
    useful_var_path = os.path.join(result_dir, 'useful_var_dic.pkl')
    with open(useful_var_path, 'wb') as f:
        pickle.dump(useful_var_dic, f)

def baseline_audio_kcenter(local_wav_list, audio_embs_dir, visual_embs_dir, result_dir, config, from_preds=False):
    """
    Baseline: CAM++ & SC + k-means clustering with visual centers
    
    对应现有的audio-vision joint clustering方法：
    1. 先进行audio-only SC和visual-only clustering
    2. 利用视觉信息对语音聚类结果进行修正
    
    Args:
        local_wav_list: List of wav file paths
        audio_embs_dir: Directory containing audio embeddings
        visual_embs_dir: Directory containing visual embeddings
        result_dir: Directory to save results
        config: Configuration object
        from_preds: If True, use predictions from model
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {current_time} Running baseline: CAM++ & SC + k-means with visual centers")
    
    # Build cluster object
    cluster = build('cluster', config)

    # Load embeddings
    alabels_embeddings_ft_path = os.path.join(result_dir, 'alabels_embeddings.pkl')
    if os.path.exists(alabels_embeddings_ft_path):
        audio_embeddings, audio_seg_ids, useful_var_dic = load_alabels_embeddings_ft(alabels_embeddings_ft_path)
        audio_times = useful_var_dic['audio_times']
        time_begin_crt_list = useful_var_dic['time_begin_crt_list']
    else:
        audio_embeddings, audio_seg_ids, _, audio_times, time_begin_crt_list = load_embeds_audio(local_wav_list, audio_embs_dir)
    
    visual_embeddings_vad, visual_times_vad = load_embed_vision_vad(local_wav_list, visual_embs_dir, time_begin_crt_list)
    
    # Visual clustering
    if os.path.exists(alabels_embeddings_ft_path):
        vlabels_vad = useful_var_dic['vlabels_vad']
    else:
        vlabels_vad = cluster.vision_cluster(visual_embeddings_vad)
        vlabels_vad = reset_cluster_ids(vlabels_vad)
        summary_cluster_results(vlabels_vad, modal_type='visual_vad_baseline_kcenter')
    
    # get alabels from predictions or sc
    suffix = '_from_model' if from_preds else ''
    if not from_preds:
        # Audio clustering
        alabels = cluster.audio_cluster(audio_embeddings)
        alabels = reset_cluster_ids(alabels)
    else:
        # Load predictions from model
        alabels = load_alabels_preds(result_dir, audio_seg_ids)
    
    summary_cluster_results(alabels, modal_type=f'audio_baseline_kcenter_before_fusion{suffix}')
    out_json = os.path.join(result_dir, f'cluster_results_audio_baseline_kcenter_before_fusion{suffix}.json')
    save_cluster_results_audio(alabels, audio_seg_ids, out_json)

    # Joint clustering: use visual information to modify audio clustering
    alabels, _ = cluster(audio_embeddings, visual_embeddings_vad, 
                                                audio_times, visual_times_vad, 
                                                config, alabels, vlabels_vad)
    
    # Save results
    summary_cluster_results(alabels, modal_type='audio_baseline_kcenter')
    out_json = os.path.join(result_dir, 'pseudo_labels_audio_baseline_kcenter.json')
    save_cluster_results_audio(alabels, audio_seg_ids, out_json)

    # Save useful variables
    useful_var_dic = {}
    useful_var_dic['audio_seg_ids'] = audio_seg_ids
    useful_var_dic['alabels'] = alabels # can be used for get mapping between old cluster ids and new cluster ids in future, don't apply now
    useful_var_dic['audio_times'] = audio_times
    useful_var_dic['time_begin_crt_list'] = time_begin_crt_list
    useful_var_dic['vlabels_vad'] = vlabels_vad
    useful_var_path = os.path.join(result_dir, 'useful_var_dic.pkl')
    with open(useful_var_path, 'wb') as f:
        pickle.dump(useful_var_dic, f)

def baseline_audio_pcc(local_wav_list, audio_embs_dir, visual_embs_dir, result_dir, config):
    """
    Baseline: CAM++ & Pairwise Constrained Clustering (PCC)
    
    使用成对约束进行聚类增强。
    
    Args:
        local_wav_list: List of wav file paths
        audio_embs_dir: Directory containing audio embeddings
        visual_embs_dir: Directory containing visual embeddings
        result_dir: Directory to save results
        config: Configuration object
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {current_time} Running baseline: CAM++ & PCC")
    
    # Build cluster object
    cluster = build('cluster', config)

    # Load embeddings
    alabels_embeddings_ft_path = os.path.join(result_dir, 'alabels_embeddings.pkl')
    if os.path.exists(alabels_embeddings_ft_path):
        audio_embeddings, audio_seg_ids, useful_var_dic = load_alabels_embeddings_ft(alabels_embeddings_ft_path)
        audio_times = useful_var_dic['audio_times']
        time_begin_crt_list = useful_var_dic['time_begin_crt_list']
    else:
        audio_embeddings, audio_seg_ids, _, audio_times, time_begin_crt_list = load_embeds_audio(local_wav_list, audio_embs_dir)
    
    visual_embeddings_vad, visual_times_vad = load_embed_vision_vad(local_wav_list, visual_embs_dir, time_begin_crt_list)

    # Visual clustering
    if os.path.exists(alabels_embeddings_ft_path):
        vlabels_vad = useful_var_dic['vlabels_vad']
    else:
        vlabels_vad = cluster.vision_cluster(visual_embeddings_vad)
        vlabels_vad = reset_cluster_ids(vlabels_vad)
        summary_cluster_results(vlabels_vad, modal_type='visual_vad_baseline_pcc')
    
    # Pairwise constrained clustering
    constrainedcluster = ConstrainedCluster(
        spectralcluster=cluster.audio_cluster.cluster, 
        alpha_v=1, 
        beta=0, 
        delta_percentile=95
    )
    alabels = constrainedcluster(audio_embeddings, audio_seg_ids, 
                                 audio_times, visual_times_vad, vlabels_vad)
    
    # Save results
    summary_cluster_results(alabels, modal_type='audio_baseline_pcc')
    out_json = os.path.join(result_dir, 'pseudo_labels_audio_baseline_pcc.json')
    save_cluster_results_audio(alabels, audio_seg_ids, out_json)
    
    # Save useful variables
    useful_var_dic = {}
    useful_var_dic['audio_seg_ids'] = audio_seg_ids
    useful_var_dic['alabels'] = alabels # can be used for get mapping between old cluster ids and new cluster ids in future, don't apply now
    useful_var_dic['audio_times'] = audio_times
    useful_var_dic['time_begin_crt_list'] = time_begin_crt_list
    useful_var_dic['vlabels_vad'] = vlabels_vad
    useful_var_path = os.path.join(result_dir, 'useful_var_dic.pkl')
    with open(useful_var_path, 'wb') as f:
        pickle.dump(useful_var_dic, f)

def baseline_joint(local_wav_list, audio_embs_dir, visual_embs_dir, result_dir, config, from_preds=False, K=10):
    """
    Baseline: CAM++ & CurricularFace & joint SC
    
    使用联合聚类方法：
    1. 对 active speaker face 和 mid-frame face 进行聚类并对齐
    2. 筛选符合条件的样本（audio segment有对应的单一mid-frame face）
    3. 对筛选后的样本进行 audio-visual joint embedding concatenation 和 spectral clustering
    4. 通过 majority voting 确定筛选样本的最终标签
    5. 使用最近邻原则扩展到所有样本
    
    Args:
        local_wav_list: List of wav file paths
        audio_embs_dir: Directory containing audio embeddings
        visual_embs_dir: Directory containing visual embeddings
        result_dir: Directory to save results
        config: Configuration object
        from_preds: If True, use predictions from model
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {current_time} Running baseline: CAM++ & CurricularFace & joint SC")
    
    # Build cluster object
    config_mf = copy.deepcopy(config)
    config_mf.vision_cluster['args']['fix_cos_thr'] = config_mf.fix_cos_thr_mf
    del config_mf.audio_cluster, config_mf.cluster
    cluster_mf = build('vision_cluster', config_mf)
    cluster = build('cluster', config)  # since config only defines original joint clustering
    joint_cluster = JointClustering_baseline(cluster.audio_cluster, cluster.vision_cluster)
    
    # ============ Step 1: Load embeddings ============
    # Load audio embeddings
    alabels_embeddings_ft_path = os.path.join(result_dir, 'alabels_embeddings.pkl')
    if os.path.exists(alabels_embeddings_ft_path):
        audio_embeddings, audio_seg_ids, useful_var_dic = load_alabels_embeddings_ft(alabels_embeddings_ft_path)
        audio_times = useful_var_dic['audio_times']
        time_begin_crt_list = useful_var_dic['time_begin_crt_list']
    else:
        audio_embeddings, audio_seg_ids, _, audio_times, time_begin_crt_list = load_embeds_audio(local_wav_list, audio_embs_dir)
    
    # Load visual embeddings for mid-frame clustering
    vlabels_mf_embeddings_ft_path = os.path.join(result_dir, 'vlabels_mf_embeddings.pkl')
    if os.path.exists(vlabels_mf_embeddings_ft_path):
        with open(vlabels_mf_embeddings_ft_path, 'rb') as f:
            visual_embeddings_dic = pickle.load(f)
        
        audio_seg_ids_mf = useful_var_dic['audio_seg_ids_mf']
        face_idxs_mf = useful_var_dic['face_idxs_mf']
        keys_mf_all = [f"{audio_seg_id_mf}_{int(face_idx)}" for audio_seg_id_mf, face_idx in zip(audio_seg_ids_mf, face_idxs_mf)]
        visual_embeddings_mf = np.array([visual_embeddings_dic[key] for key in keys_mf_all])
    else:
        visual_embeddings_mf, audio_seg_ids_mf, face_idxs_mf = load_embeds_vision_mf(local_wav_list, visual_embs_dir)

    if not from_preds:
        # ============ Step 2: Visual clustering ============
        # Active speaker face clustering
        visual_embeddings_vad, visual_times_vad = load_embed_vision_vad(local_wav_list, visual_embs_dir, time_begin_crt_list)
        vlabels_vad = cluster.vision_cluster(visual_embeddings_vad)
        vlabels_vad = reset_cluster_ids(vlabels_vad)
        summary_cluster_results(vlabels_vad, modal_type='visual_vad_baseline_joint')
        vad_json_path = os.path.join(result_dir, 'cluster_results_vision_vad.json')
        save_cluster_results_vision_vad(audio_times, visual_times_vad, audio_seg_ids, vlabels_vad, vad_json_path)
        
        # Mid-frame face clustering
        vlabels_mf = cluster_mf(visual_embeddings_mf)
        vlabels_mf = reset_cluster_ids(vlabels_mf)
        summary_cluster_results(vlabels_mf, modal_type='visual_mid_frame_before_vision_align')
        save_cluster_results_vision_mf(vlabels_mf, audio_seg_ids_mf, face_idxs_mf, 
                                      os.path.join(result_dir, 'cluster_results_faces_mid_frame_before_vision_align.json'))
        
        # Align mid-frame face clustering results with active speaker face clustering results
        ### unaligned_label is not important, since joint clusteringonly cares aligned samples
        align_cos_thr = 0.5
        print(f"[INFO] Set cos-similarity threshold to {align_cos_thr} during aligning mid-frame faces clustering and active speaker face clustering.")
        vlabels_mf = align_clusters2clusters(copy.deepcopy(vlabels_mf), copy.deepcopy(vlabels_vad), 
                                            visual_embeddings_mf, visual_embeddings_vad, 
                                            align_cos_thr=align_cos_thr, unaligned_label=-3)
        summary_cluster_results(vlabels_mf, modal_type='visual_mid_frame_vision_aligned')
        save_cluster_results_vision_mf(vlabels_mf, audio_seg_ids_mf, face_idxs_mf, 
                                      os.path.join(result_dir, 'cluster_results_faces_mid_frame_vision_aligned.json'))
        
        # ============ Step 3: Load vad clustering results to filter samples ============
        # Read saved cluster_results_vision_vad.json
        with open(vad_json_path.replace('.json', '_major.json'), 'r') as f:
            vad_cluster_results = json.load(f)
        
        # Filter mid-frame samples
        ## Condition1: audio_seg_ids_mf in vad_cluster_results keys AND vlabels_mf matches value
        visual_mf_selected_index = []
        for i in range(len(audio_seg_ids_mf)):
            seg_id = audio_seg_ids_mf[i]
            if seg_id in vad_cluster_results and vlabels_mf[i] == vad_cluster_results[seg_id]:
                    visual_mf_selected_index.append(i)
        visual_mf_selected_index = np.array(visual_mf_selected_index)
        ## Condition2: verify that only one mid-frame face is selected for each audio segment
        unique_ids, counts = np.unique(audio_seg_ids_mf[visual_mf_selected_index], return_counts=True)
        selected_mf_seg_ids = unique_ids[counts == 1]
        visual_mf_selected_index = np.array([i for i in visual_mf_selected_index if audio_seg_ids_mf[i] in selected_mf_seg_ids])
        print(f"[INFO] Selected {len(visual_mf_selected_index)} mid-frame visual samples matching VAD clustering")
        
        # ============ Step 4: Audio clustering and filter corresponding samples ============
        alabels = cluster.audio_cluster(audio_embeddings)
        alabels = reset_cluster_ids(alabels)
        summary_cluster_results(alabels, modal_type='audio_baseline_joint_before_fusion')
        save_cluster_results_audio(alabels, audio_seg_ids, 
                                   os.path.join(result_dir, 'cluster_results_audio_baseline_joint_before_fusion.json'))
        
        # Filter audio samples: audio_seg_ids in audio_seg_ids_mf[visual_mf_selected_index]
        audio_selected_index = np.where(np.isin(audio_seg_ids, list(selected_mf_seg_ids)))[0]
        print(f"[INFO] Selected {len(audio_selected_index)} audio samples matching selected mid-frame faces")
        
        # Verify that both have the same number of samples
        selected_audio_seg_ids = audio_seg_ids[audio_selected_index]
        visual_mf_selected_index = np.array([i for i in visual_mf_selected_index if audio_seg_ids_mf[i] in selected_audio_seg_ids])
        assert len(audio_selected_index) == len(visual_mf_selected_index), \
            f"Number of selected audio ({len(audio_selected_index)}) and visual ({len(visual_mf_selected_index)}) samples must match!"
    else:
        # ============ Load mode: from_preds=True ============
        print(f"[INFO] Loading predictions from fine-tuned model")
        # Load useful variables
        visual_mf_selected_index = useful_var_dic['visual_mf_selected_index']
        audio_selected_index = useful_var_dic['audio_selected_index']
        
        # Load speaker predictions and face predictions
        alabels = load_alabels_preds(result_dir, audio_seg_ids)
        
        # Load face predictions
        vlabels_mf_pred_dic_path = os.path.join(result_dir, 'vlabels_mf_pred_dic.pkl')
        assert os.path.exists(vlabels_mf_pred_dic_path), f"vlabels_mf_pred_dic.pkl must exist in {result_dir} when from_preds=True"
        with open(vlabels_mf_pred_dic_path, 'rb') as f:
            vlabels_mf_pred_dic = pickle.load(f)
        keys_mf_all = [f"{audio_seg_id_mf}_{int(face_idx)}" for audio_seg_id_mf, face_idx in zip(audio_seg_ids_mf, face_idxs_mf)]
        vlabels_mf = np.array([vlabels_mf_pred_dic[key] for key in keys_mf_all])
        
    # ============ Step 5: Joint clustering using JointClustering_baseline ============
    # K: Number of main clusters to keep (others marked as -1), default 10
    alabels_final, vlabels_mf_final, joint_labels_dic = joint_cluster(
        audio_embeddings, visual_embeddings_mf, 
        audio_seg_ids, audio_seg_ids_mf,
        audio_selected_index, visual_mf_selected_index,
        alabels, vlabels_mf, K=2*config.main_actors_num
    )
    
    # ============ Step 6: Save results ============
    # Save audio pseudo labels
    summary_cluster_results(alabels_final, modal_type='audio_joint_final')
    out_json_audio = os.path.join(result_dir, 'pseudo_labels_audio_baseline_joint.json')
    save_cluster_results_audio(alabels_final, audio_seg_ids, out_json_audio)
    
    # Save visual mid-frame pseudo labels (all samples)
    summary_cluster_results(vlabels_mf_final, modal_type='visual_mf_joint_final')
    out_json_visual = os.path.join(result_dir, 'pseudo_labels_faces_mid_frame_all_baseline_joint.json')
    save_cluster_results_vision_mf(vlabels_mf_final, audio_seg_ids_mf, face_idxs_mf, out_json_visual)
    shutil.copy(out_json_visual, os.path.join(result_dir, 'pseudo_labels_faces_mid_frame_train_baseline_joint.json'))
    
    # Save joint labels for selected samples
    out_json_joint = os.path.join(result_dir, 'pseudo_labels_joint_selected.json')
    with open(out_json_joint, 'w') as f:
        json.dump(joint_labels_dic, f, indent=2)
    
    # Save useful variables
    useful_var_dic = {}
    useful_var_dic['audio_seg_ids'] = audio_seg_ids
    useful_var_dic['audio_times'] = audio_times
    useful_var_dic['time_begin_crt_list'] = time_begin_crt_list
    useful_var_dic['audio_seg_ids_mf'] = audio_seg_ids_mf
    useful_var_dic['face_idxs_mf'] = face_idxs_mf
    useful_var_dic['visual_mf_selected_index'] = visual_mf_selected_index
    useful_var_dic['audio_selected_index'] = audio_selected_index
    useful_var_path = os.path.join(result_dir, 'useful_var_dic.pkl')
    with open(useful_var_path, 'wb') as f:
        pickle.dump(useful_var_dic, f)
    print(f"[INFO] Saved useful variables to {useful_var_path}")

def main():
    args = parser.parse_args()
    
    # Get wav list
    with open(args.wavs, 'r') as f:
        wav_list = [i.strip() for i in f.readlines()]
    wav_list.sort()
    
    os.makedirs(args.result_dir, exist_ok=True)
    print("[INFO] Start clustering...")
    
    # Load config
    config = build_config(args.conf)
    
    # Run baseline method
    if args.baseline_method == 'sc':
        if hasattr(config, 'audio_cluster') and hasattr(config, 'vision_cluster'):
            config.cluster = config.audio_cluster
            del config.audio_cluster, config.vision_cluster
        baseline_audio_sc(wav_list, args.audio_embs_dir, args.result_dir, config)
    elif args.baseline_method == 'vbx':
        if hasattr(config, 'audio_cluster') and hasattr(config, 'vision_cluster'):
            config.cluster = config.audio_cluster
            del config.audio_cluster, config.vision_cluster
        baseline_audio_vbx(wav_list, args.audio_embs_dir, args.result_dir, config, args.from_preds)
    elif args.baseline_method == 'kcenter':
        assert args.visual_embs_dir is not None, "--visual_embs_dir required for kcenter method"
        baseline_audio_kcenter(wav_list, args.audio_embs_dir, args.visual_embs_dir, 
                              args.result_dir, config, args.from_preds)
    elif args.baseline_method == 'pcc':
        assert args.visual_embs_dir is not None, "--visual_embs_dir required for pcc method"
        baseline_audio_pcc(wav_list, args.audio_embs_dir, args.visual_embs_dir, 
                          args.result_dir, config)
    elif args.baseline_method == 'joint':
        assert args.visual_embs_dir is not None, "--visual_embs_dir required for joint method"
        baseline_joint(wav_list, args.audio_embs_dir, args.visual_embs_dir, 
                      args.result_dir, config, args.from_preds)
    else:
        raise ValueError(f"Unknown baseline method: {args.baseline_method}")
    
    print(f"[INFO] Baseline method {args.baseline_method} completed!")


if __name__ == "__main__":
    main()
