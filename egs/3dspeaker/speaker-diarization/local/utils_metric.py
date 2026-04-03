import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

###############
# utils for compute accuracy
###############

main_character_list_IL = ['傅老', '和平', '志新', '志国', '圆圆', '小凡', '小张', '燕红']
main_character_list_BB = ['Sheldon', 'Leonard', 'Penny', 'Howard', 'Raj']
main_character_list = main_character_list_IL + main_character_list_BB

def eval_test_split(xlsx_path):
    # 根据 speaker 和 duration 分层抽样,获取 20% 的 keys
    ## 筛选有标注的数据
    df = pd.read_excel(xlsx_path)
    df = df[df['whether annotate speaker'] == 'Yes']
    ## 提取 keys, speakers, durations
    keys = df.apply(lambda row: f"E{int(row['Episode']):02}-{int(row['Text Index'])}", axis=1)
    speakers = df['speaker']
    speaker_labels = ['Others' if speaker not in main_character_list else speaker for speaker in speakers] # Replace all non-main characters with 'Others'
    # durations = df.apply(lambda row: time_to_seconds(row['End Time']) - time_to_seconds(row['Start Time']), axis=1)
    # bins = [0, 1, 2, 3, 4, float('inf')]
    # duration_groups = (np.digitize(durations, bins) - 1).tolist()  # 取值范围 [0, 4]，len(group_indices) == len(durations)
    # Stratified split
    # stratify_labels = [f"{spk}_{grp}" for spk, grp in zip(speaker_labels, duration_groups)]
    _, valid_keys = train_test_split(keys, test_size=0.2, stratify=speaker_labels, random_state=100)
    valid_keys_list = valid_keys.tolist()
    return valid_keys_list

def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

def name2onehot(name, name2idx):
    arr = np.zeros(len(name2idx), dtype=int)
    idx = name2idx.get(name, name2idx['Others'])
    arr[idx] = 1
    return arr

# given a list of 1's index, convert it to 0-1 encoding
def list2onehot(lst, n_states):
  if type(lst) == int:
    lst_new = [lst]
  elif type(lst) == list:
    lst_new = lst
  else:
    raise ValueError('Input should be a list or an integer')
  onehot = list(map(lambda x: 1 if x in lst_new else 0, range(n_states)))
  return onehot

def class_matching(onehot_ref, onehot_pred, others_chara_id=None):
    """
    支持 onehot_ref.shape[1] != onehot_pred.shape[1] 的情况。此时效果相当于，仅建立部分簇/标注的映射关系。
    返回 聚类簇id 到 character id(face/speaker) 的映射字典。
    """
    # 检查输入形状
    assert onehot_ref.ndim == 2 and onehot_pred.ndim == 2, "Inputs must be 2D arrays."
    assert onehot_ref.shape[0] == onehot_pred.shape[0], "Number of samples must be the same."
    n_samples = onehot_ref.shape[0]
    # 将onehot编码转换为标签
    label_ref = np.argmax(onehot_ref, axis=1)
    label_pred = np.argmax(onehot_pred, axis=1)
    # ensure classes are sorted in ascending order
    ref_classes = np.sort(np.unique(label_ref))
    pred_classes = np.sort(np.unique(label_pred))
    n_ref, n_pred = len(ref_classes), len(pred_classes)
    ref_classes2idx = {cls: idx for idx,cls in enumerate(ref_classes)}
    pred_classes2idx = {cls: idx for idx,cls in enumerate(pred_classes)}
    # 构建计数矩阵
    count_matrix = np.zeros((n_ref, n_pred), dtype=int)
    for i in range(n_samples):
        label_idx_ref = ref_classes2idx[label_ref[i]]
        label_idx_pred = pred_classes2idx[label_pred[i]]
        count_matrix[label_idx_ref, label_idx_pred] += 1

    # 委托给 get_map 计算映射
    others_chara_idx = ref_classes2idx.get(others_chara_id, None) if others_chara_id is not None else None
    row_ind, col_ind = get_map(count_matrix, others_chara_idx)
    # 构建映射字典：pred_class -> ref_class。考虑了n_ref < n_pred时，label_ref中不存在others，但是部分簇需要映射到others的情况
    ref_idx2class = {idx: cls for cls, idx in ref_classes2idx.items()}
    pred_idx2class = {idx: cls for cls, idx in pred_classes2idx.items()}
    mapping = {pred_idx2class[col]: ref_idx2class[row] if row in ref_idx2class else others_chara_id for row, col in zip(row_ind, col_ind)}
    return mapping


def get_map(count_matrix, others_chara_idx=None):
    """
    给定计数矩阵，使用匈牙利算法返回最佳映射。
    mapping: pred_class_idx -> ref_class_idx (当需要映射到 Others 时，值为 others_chara_id)
    """
    n_ref, n_pred = count_matrix.shape[0], count_matrix.shape[1]

    # 匈牙利算法分配
    if n_ref == n_pred:
        row_ind, col_ind = linear_sum_assignment(-count_matrix)
    elif n_ref > n_pred:
        # 列补齐
        pad_val = np.min(count_matrix) - 1
        padded_matrix = np.pad(count_matrix, ((0,0),(0,n_ref-n_pred)), constant_values=pad_val)
        row_ind, col_ind = linear_sum_assignment(-padded_matrix)
        valid = col_ind < n_pred
        row_ind = row_ind[valid]
        col_ind = col_ind[valid]
    else:
        assert others_chara_idx is not None, "When n_ref < n_pred, others_chara_id must be provided."
        # 行补齐
        pad_val = np.min(count_matrix) - 1
        padded_matrix = np.pad(count_matrix, ((0,n_pred-n_ref),(0,0)), constant_values=pad_val)
        row_ind, col_ind = linear_sum_assignment(-padded_matrix)
        valid = row_ind < n_ref
        row_ind_valid = row_ind[valid]        
        col_ind_valid = col_ind[valid]
        col_ind_others = col_ind[~valid]
        if len(col_ind_others) > 0:
            row_ind_others = np.array([others_chara_idx]*len(col_ind_others))
            row_ind = np.concatenate((row_ind_valid, row_ind_others))
            col_ind = np.concatenate((col_ind_valid, col_ind_others))
        else:
            row_ind = row_ind_valid
            col_ind = col_ind_valid        

    return row_ind, col_ind


# calculate accuracy given one-hot encoded label and prediction for multi-class classification
def cal_accuracy_onehot(label_onehot, pred_onehot):
    assert label_onehot.shape == pred_onehot.shape, "Shape of label and prediction must be the same."
    n_samples = label_onehot.shape[0]
    correct = np.sum(np.all(label_onehot == pred_onehot, axis=1))
    accuracy = correct / n_samples if n_samples > 0 else 0.0
    return round(accuracy, 4)


def cal_clustering_metrics(ref_labels, pred_labels):
    """
    计算整体聚类指标:
    - NMI
    - ARI
    - mean entropy per cluster
    - mean maximal purity per cluster
    """
    assert len(ref_labels) == len(pred_labels), "ref_labels and pred_labels must have the same length."
    if len(ref_labels) == 0:
        return {
            'overall_nmi': 0.0,
            'overall_ari': 0.0,
            'overall_mean_entropy_per_cluster': 0.0,
            'overall_mean_maximal_purity_per_cluster': 0.0,
        }

    ref_classes = sorted(set(ref_labels))
    pred_classes = sorted(set(pred_labels))
    ref_to_idx = {label: idx for idx, label in enumerate(ref_classes)}
    pred_to_idx = {label: idx for idx, label in enumerate(pred_classes)}

    ref_ids = np.array([ref_to_idx[label] for label in ref_labels], dtype=int)
    pred_ids = np.array([pred_to_idx[label] for label in pred_labels], dtype=int)

    nmi = normalized_mutual_info_score(pred_ids, ref_ids, average_method='arithmetic')
    ari = adjusted_rand_score(pred_ids, ref_ids)

    entropies = []
    purities = []
    for pred_id in np.unique(pred_ids):
        of_this_cluster = (pred_ids == pred_id)
        if of_this_cluster.sum() == 0:
            continue
        _, counts = np.unique(ref_ids[of_this_cluster], return_counts=True)
        probs = counts / counts.sum()
        entropies.append(float(entropy(probs)))
        purities.append(float(counts.max() / counts.sum()))

    return {
        'overall_nmi': round(float(nmi), 4),
        'overall_ari': round(float(ari), 4),
        'overall_mean_entropy_per_cluster': round(float(np.mean(entropies)), 4),
        'overall_mean_maximal_purity_per_cluster': round(float(np.mean(purities)), 4),
    }



###############
# utils for compute EER
###############
import random
from operator import itemgetter
from sklearn import metrics
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