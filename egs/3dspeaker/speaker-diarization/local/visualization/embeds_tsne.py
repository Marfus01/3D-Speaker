import numpy as np
import pandas as pd
import os, pickle, random
from operator import itemgetter
from sklearn import manifold, metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_embeddings(TV_name, modal, load_from_pretrain=False, round_idx=0):
    if load_from_pretrain:
        dat_dir = f'/data/home/scv7387/run/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-diarization/runs/{TV_name}/exp_video'
        if modal == 'audio':
            embeds_arr, embeds_sample_ids = load_embeddings_pretrained_audio(audio_embs_dir=f'{dat_dir}/embs')
        elif modal == 'face':
            embeds_arr, embeds_sample_ids = load_embeddings_pretrained_mf(mf_embs_dir=f'{dat_dir}/embs_video')
        else:
            raise ValueError(f'Unknown modal: {modal}')
    else:
        if modal == 'audio':
            file_name = 'alabels_embeddings.pkl'
        elif modal == 'face':
            file_name = 'vlabels_mf_embeddings.pkl'
        else:
            raise ValueError(f'Unknown modal: {modal}')

        dat_dir = f"/data/home/scv7387/run/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-diarization/runs/{TV_name}/exp_video/result/self_supervised/2. 仅微调dense layer，从hid_feat构建数据集/exp27.7"
        embeds_arr, embeds_sample_ids = load_embeddings_ft(file_path=os.path.join(dat_dir, f'round{round_idx}', 'pseudo_label', file_name))
    return embeds_arr, embeds_sample_ids

def load_embeddings_pretrained_audio(audio_embs_dir):
    audio_embeddings = np.array([], dtype=np.float32)
    audio_seg_ids = np.array([], dtype='<U50')  # of the same length as audio_embeddings
    
    file_names = os.listdir(audio_embs_dir)
    file_names.sort()
    print(f'Loading audio embeddings from the following files: {file_names}')
    
    for file_idx, file_name in enumerate(file_names):
        with open(os.path.join(audio_embs_dir, file_name), 'rb') as f:
            stat_obj = pickle.load(f)
        if file_idx == 0:
            audio_embeddings = stat_obj['embeddings']
            audio_seg_ids = stat_obj['subseg_ids']
        else:
            audio_embeddings = np.vstack((audio_embeddings, stat_obj['embeddings']))
            audio_seg_ids = np.hstack((audio_seg_ids, stat_obj['subseg_ids']))
    return audio_embeddings, audio_seg_ids

def load_embeddings_pretrained_mf(mf_embs_dir):
    visual_embeddings_mf = np.array([], dtype=np.float32)
    audio_seg_ids_mf = np.array([], dtype='<U50')
    face_idxs_mf = np.array([], dtype=np.int32)

    file_names = [f for f in os.listdir(mf_embs_dir) if f.endswith('_midframe.pkl')]
    file_names.sort()
    print(f'Loading mid-frame visual embeddings from the following files: {file_names}')

    for file_idx, file_name in enumerate(file_names):
        with open(os.path.join(mf_embs_dir, file_name), 'rb') as f:
            stat_obj = pickle.load(f)
        if file_idx == 0:
            visual_embeddings_mf = stat_obj['feat']
            audio_seg_ids_mf = stat_obj['audio_seg_id'] # np.ndarray, (N, )
            face_idxs_mf = stat_obj['face_idx'] # np.ndarray, (N, )
        else:
            visual_embeddings_mf = np.vstack((visual_embeddings_mf, stat_obj['feat']))
            audio_seg_ids_mf = np.hstack((audio_seg_ids_mf, stat_obj['audio_seg_id']))
            face_idxs_mf = np.hstack((face_idxs_mf, stat_obj['face_idx']))

    face_ids_mf = np.array([f"{seg_id}_{int(face_idx)}" for seg_id, face_idx in zip(audio_seg_ids_mf, face_idxs_mf)])
    return visual_embeddings_mf, face_ids_mf

def load_embeddings_ft(file_path):
    """
    Load embeddings from .pkl file.

    Returns:
        arr: ndarray of shape (N, dim)
        sample_ids: list of ids if the source was a dict (preserve insertion order), else None
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    embeds_arr = np.vstack([np.asarray(v) for v in list(obj.values())])
    embeds_sample_ids = np.array(list(obj.keys()))
    return embeds_arr, embeds_sample_ids

def get_labels(TV_name, modal, characters_index_dic):
    if modal == 'audio':
        xls_file = f'/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/annotation/text_annotated.xlsx'
        df = pd.read_excel(xls_file, sheet_name='Sheet1')
        df = df[df['whether annotate speaker'] == 'Yes']
        keys = df.apply(lambda row: f"E{int(row['Episode']):02}-{int(row['Text Index'])}", axis=1)  # 与聚类结果中的 segment ID 完全对应
        labels = np.array([characters_index_dic.get(speaker, characters_index_dic['Others']) for speaker in df['speaker'].tolist()])
    
    elif modal == 'face':
        xls_file = f'/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/annotation/faces_annotation_with_loc_new.xlsx'
        df = pd.read_excel(xls_file, sheet_name='Sheet1')
        keys = df.apply(lambda row: f"{row['audio_seg_id']}_{int(row['face index'])}", axis=1)  # 与聚类结果中的 face ID 完全对应
        labels = np.array([characters_index_dic.get(face_name, characters_index_dic['Others']) for face_name in df['face label'].tolist()])
    else:
        raise ValueError('Modal not recognized')
    
    return labels, keys

def save_testEER(labels, keys, modal, save_path, characters_index_dic):
    random.seed(100)  # Set random seed for reproducibility
    characters_list = list(characters_index_dic.keys())
    characters_index_dic_reverse = {v: k for k, v in characters_index_dic.items()}
    # total number of positive and negative pairs
    positive_data = 3500
    negative_data = 3500

    keys_index = list(range(len(keys)))
    sim_label_list, key1_list, key2_list = [], [], []
    # num of audio/face for each character
    pos_cnt_dict = {char: 0 for char in characters_list}
    neg_cnt_dict = {char: 0 for char in characters_list}
    i = 0
    while i < positive_data: 
        random.shuffle(keys_index)
        j = 1
        while labels[keys_index[0]] != labels[keys_index[j]]:
            j += 1
        sim_label_list.append(1)
        pos_cnt_dict[characters_index_dic_reverse[int(labels[keys_index[0]])]] += 2
        key1_list.append(keys[keys_index[0]])
        key2_list.append(keys[keys_index[j]])
        i += 1

    i = 0
    while i < negative_data: 
        random.shuffle(keys_index)
        j = 1
        while labels[keys_index[0]] == labels[keys_index[j]]:
            j += 1
        sim_label_list.append(0)
        neg_cnt_dict[characters_index_dic_reverse[int(labels[keys_index[0]])]] += 1
        neg_cnt_dict[characters_index_dic_reverse[int(labels[keys_index[j]])]] += 1
        key1_list.append(keys[keys_index[0]])
        key2_list.append(keys[keys_index[j]])
        i += 1

    print(f'num of {modal} for each character in positive pairs:', pos_cnt_dict)
    print(f'num of {modal} for each character in negative pairs:', neg_cnt_dict)
    print('total num of positive pairs:', sum(pos_cnt_dict.values())/2)
    print('total num of negative pairs:', sum(neg_cnt_dict.values())/2)

    with open(save_path, 'w') as f:
        for i,j,k in zip(sim_label_list, key1_list, key2_list):
            f.write(str(i)+' '+j+' '+k+'\n')

def read_testEER(file_path):
    sim_label_list, key1_list, key2_list = [], [], []
    for line in open(file_path).read().splitlines():
        items = line.split()
        sim_label_list.append(int(items[0]))
        key1_list.append(items[1])
        key2_list.append(items[2])
    return sim_label_list, key1_list, key2_list

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr))) # np.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    return tunedThreshold, eer, fpr, fnr

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):
      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def evaluate_EER(embeds_dict, testEER_file):
    sim_label_list, key1_list, key2_list = read_testEER(testEER_file)
    scores = []
    for k1, k2 in zip(key1_list, key2_list):
        emb1 = embeds_dict[k1]
        emb2 = embeds_dict[k2]
        score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        scores.append(score)
    eer = tuneThresholdfromScore(scores, sim_label_list, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, sim_label_list)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    return eer, minDCF



TV_name = 'I love my family' # 'I love my family', 'the big bang theory'
modal = 'face' # 'face', 'audio'
load_from_pretrain = False  # whether to load embeddings from pre-trained model or fine-tuned model
round_idx = 9  # only used when load_from_pretrain is False

if TV_name == 'I love my family':
    characters_index_dic = {'傅老': 0, '和平': 1, '志新': 2, '志国': 3, '圆圆': 4, '小凡': 5, '小张': 6, '燕红': 7, 'Others': 8}
elif TV_name == 'the big bang theory':
    characters_index_dic = {'Sheldon': 0, 'Leonard': 1, 'Penny': 2, 'Howard': 3, 'Raj': 4, 'Others': 5}
else:
    raise ValueError('TV name not recognized')

# load embeddings and labels
embeds_arr, embeds_sample_ids = load_embeddings(TV_name, modal, load_from_pretrain, round_idx)
embeds_arr = embeds_arr / np.linalg.norm(embeds_arr, axis=1, keepdims=True)
print('Loaded embeddings shape:', embeds_arr.shape)
labels, useful_keys = get_labels(TV_name, modal, characters_index_dic)
useful_indexs = np.array([embeds_sample_ids.tolist().index(key) for key in useful_keys])
print(f'Number of {modal} samples with label: {len(labels)}')

# save testEER file(if not exist), and evaluate EER and minDCF
testEER_file =f'/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/annotation/{modal}_testEER.txt'
if not os.path.exists(testEER_file):
    save_testEER(labels, useful_keys, modal, testEER_file, characters_index_dic)
embeds_dict = {embeds_sample_ids[i]: embeds_arr[i] for i in useful_indexs}
eer, minDCF = evaluate_EER(embeds_dict, testEER_file)
print(f'EER: {eer:.2f}%, minDCF: {minDCF:.4f}')

# t-SNE visualization
## use all embeds to get tsne, but only plot labeled ones
embeds_arr_2dim = manifold.TSNE(n_components=2,
                                random_state=0,
                                init='random',
                                learning_rate='auto').fit_transform(embeds_arr)
embeds_arr_2dim = embeds_arr_2dim[useful_indexs]

## Define a custom order for sorting
if TV_name == 'the big bang theory':
    legend_order = ['Sheldon', 'Leonard', 'Penny', 'Howard', 'Rajesh', 'Others']  
else:
    legend_order = ['Fulao', 'Heping', 'Zhixin', 'Zhiguo', 'Yuanyuan', 'Xiaofan', 'Xiaozhang', 'Yanhong', 'Others']
## Create legend handles in the specified order
handles = []
color_list = np.vstack((np.array(plt.get_cmap('tab20').colors), plt.get_cmap('tab20c').colors[-3]))  # Add a color for 'Others'
for i in range(len(legend_order)):
    if legend_order[i] == 'Others':
        handles.append(mpatches.Patch(color=color_list[20], label=legend_order[i], alpha=0.5))
    else:
        handles.append(mpatches.Patch(color=color_list[i], label=legend_order[i], alpha=0.5))

## Plotting
point_size=20
colors = list(map(lambda x: color_list[x] if x < max(labels) else color_list[20], labels))
fig, ax = plt.subplots(figsize=(5, 5))  # Set the figure's height and width equal
ax.set_xlim(min(embeds_arr_2dim[:, 0]) - 5, max(embeds_arr_2dim[:, 0]) + 5)
ax.set_ylim(min(embeds_arr_2dim[:, 1]) - 5, max(embeds_arr_2dim[:, 1]) + 5)
ax.scatter(embeds_arr_2dim[:, 0], embeds_arr_2dim[:, 1], color=colors, alpha=0.5, s=point_size)
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.xticks(np.arange(int((min(embeds_arr_2dim[:, 0]) - 5)/25)*25, int((max(embeds_arr_2dim[:, 0]) + 5)/25+1)*25, 25))
plt.yticks(np.arange(int((min(embeds_arr_2dim[:, 1]) - 5)/25)*25, int((max(embeds_arr_2dim[:, 1]) + 5)/25+1)*25, 25))
# plt.legend(handles=handles,fontsize=15)
# plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=15)
plt.tick_params(labelsize=10)

## Save figure
save_dir = f'/data/home/scv7387/run/tv_series_plus/3D-Speaker/docs/figs/tsne_v2_all/{TV_name}'
os.makedirs(save_dir, exist_ok=True)
exp_type = 'pretrain' if load_from_pretrain else f'ft_round{round_idx}'
plt.savefig(f'{save_dir}/{modal}_{exp_type}(s={point_size}).png', bbox_inches='tight')