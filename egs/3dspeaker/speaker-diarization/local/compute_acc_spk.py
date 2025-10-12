import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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

def class_matching(onehot_ref, onehot_pred, others_spk_id=None):
    """
    支持 onehot_ref.shape[1] != onehot_pred.shape[1] 的情况。此时效果相当于，仅建立部分簇/标注的映射关系。
    返回 聚类簇id 到 speaker id 的映射字典。
    """
    # 检查输入形状
    assert onehot_ref.ndim == 2 and onehot_pred.ndim == 2, "Inputs must be 2D arrays."
    assert onehot_ref.shape[0] == onehot_pred.shape[0], "Number of samples must be the same."
    n_samples = onehot_ref.shape[0]
    # 将onehot编码转换为标签
    label_ref = np.argmax(onehot_ref, axis=1)
    label_pred = np.argmax(onehot_pred, axis=1)
    ref_classes = np.unique(label_ref)
    pred_classes = np.unique(label_pred)
    n_ref = len(ref_classes)
    n_pred = len(pred_classes)
    # 构建计数矩阵
    count_matrix = np.zeros((n_ref, n_pred), dtype=int)
    for i in range(n_samples):
        count_matrix[label_ref[i], label_pred[i]] += 1
    
    # 匈牙利算法分配
    if n_ref == n_pred:
        row_ind, col_ind = linear_sum_assignment(-count_matrix)
    elif n_ref > n_pred:
        # 列补齐
        pad_val = np.max(count_matrix) + 1
        padded_matrix = np.pad(count_matrix, ((0,0),(0,n_ref-n_pred)), constant_values=pad_val)
        row_ind, col_ind = linear_sum_assignment(-padded_matrix)
        valid = col_ind < n_pred
        row_ind = row_ind[valid]
        col_ind = col_ind[valid]
    else:
        assert others_spk_id is not None and others_spk_id in ref_classes, "When n_ref < n_pred, others_spk_id must be provided and exist in reference classes."
        # 行补齐
        pad_val = np.max(count_matrix) + 1
        padded_matrix = np.pad(count_matrix, ((0,n_pred-n_ref),(0,0)), constant_values=pad_val)
        row_ind, col_ind = linear_sum_assignment(-padded_matrix)
        valid = row_ind < n_ref
        row_ind_valid = row_ind[valid]        
        col_ind_valid = col_ind[valid]
        col_ind_others = col_ind[~valid]
        if len(col_ind_others) > 0:
            row_ind_others = np.array([others_spk_id]*len(col_ind_others))
            row_ind = np.concatenate((row_ind_valid, row_ind_others))
            col_ind = np.concatenate((col_ind_valid, col_ind_others))
        else:
            row_ind = row_ind_valid
            col_ind = col_ind_valid        

    # 构建映射字典：pred_class -> ref_class
    mapping = {pred_classes[col]: ref_classes[row] for row, col in zip(row_ind, col_ind)}
    return mapping


# calculate accuracy given one-hot encoded label and prediction for multi-class classification
def cal_accuracy_onehot(label_onehot, pred_onehot):
    assert label_onehot.shape == pred_onehot.shape, "Shape of label and prediction must be the same."
    n_samples = label_onehot.shape[0]
    correct = np.sum(np.all(label_onehot == pred_onehot, axis=1))
    accuracy = correct / n_samples if n_samples > 0 else 0.0
    return round(accuracy, 4)

def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    json_files = [os.path.join(args.result_dir, f) for f in os.listdir(args.result_dir) if f.endswith('.json') and os.path.isfile(os.path.join(args.result_dir, f))]  # all cluster json files for evaluation
    if not json_files:
        print("No cluster json files found in", args.result_dir)
        sys.exit(1)
    
    for json_file in json_files:
        # 1. 读取聚类结果
        with open(json_file, 'r', encoding='utf-8') as f:
            cluster_dic = json.load(f)
        # 2. 读取标注文件，并筛选有标注的数据，提取对应的segment id，说话人和时长
        df = pd.read_excel(args.ref_xlsx) # Text Index列从 0 开始
        df = df[df['whether annotate speaker'] == 'Yes']
        keys = df.apply(lambda row: f"E{int(row['Episode']):02}-{int(row['Text Index'])}", axis=1)  # 与聚类结果中的 segment ID 完全对应
        speakers = df['speaker'].tolist()
        speakers = ['Others' if speaker == '小凡' else speaker for speaker in speakers] # Replace all '小凡' in speakers with 'Others'
        durations = df.apply(lambda row: time_to_seconds(row['End Time']) - time_to_seconds(row['Start Time']), axis=1)

        # 3. 获取所有有标注数据的聚类标签
        cluster_labels = [cluster_dic.get(k, -1) for k in keys]
        valid_idx = [i for i, c in enumerate(cluster_labels) if c != -1]
        if len(valid_idx) < len(keys):
          missing_keys = [keys[i] for i in range(len(keys)) if i not in valid_idx]
          print("The following keys are missing in the cluster dictionary:", missing_keys)
          keys = [keys[i] for i in valid_idx]
          speakers = [speakers[i] for i in valid_idx]
          cluster_labels = [cluster_labels[i] for i in valid_idx]
          durations = [durations[i] for i in valid_idx]

        # 4. 调整 cluster_labels，使其从 0 开始连续编号
        print('Unique original cluster ids:', set(cluster_labels))
        ## Create a mapping from unique cluster labels to consecutive integers
        unique_cluster_labels = sorted(set(cluster_labels))
        cluster_label_mapping = {label: idx for idx, label in enumerate(unique_cluster_labels)}
        ## Map cluster_labels to consecutive integers
        cluster_labels = [cluster_label_mapping[label] for label in cluster_labels]

        # 5. 根据标注文件，构建说话人名->speaker id的字典
        unique_speakers = sorted(set(speakers + ['Others']))
        name2idx = {name: idx for idx, name in enumerate(unique_speakers)}
        print("Speaker to index mapping:", name2idx)

        # 6. 将 prediction 和 label 都转为 one-hot，随后利用匈牙利算法进行标签对齐
        speaker_onehot = np.array([name2onehot(name, name2idx) for name in speakers])
        n_clusters = max(cluster_labels) + 1
        cluster_onehot = np.array(list(map(lambda x: list2onehot(x, n_clusters), cluster_labels)))
        mapping = class_matching(speaker_onehot, cluster_onehot, others_spk_id=name2idx['Others'])
        cluster_pred = np.eye(len(name2idx))[np.array([mapping[label] for label in cluster_labels])]
        print("Cluster_id to speaker_id mapping:", mapping)

        # 7. 按时长分组，计算分组/整体的 accuracy
        bins = [0, 1, 2, 3, 4, float('inf')]
        group_indices = np.digitize(durations, bins) - 1  # 取值范围 [0, 4]，len(group_indices) == len(durations)
        results = {}
        results['overall_accuracy'] = cal_accuracy_onehot(speaker_onehot, cluster_pred)
        for i in range(5):
            idx = [j for j, g in enumerate(group_indices) if g == i]
            if idx:
                acc = cal_accuracy_onehot(speaker_onehot[idx], cluster_pred[idx])
                results[f'group_{i}_accuracy'] = acc

        # 8. 保存结果
        filename = os.path.basename(json_file).replace('.json', '_accuracy.txt')
        with open(os.path.join(args.result_dir, filename), 'w') as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")

        print("Accuracy results saved to", os.path.join(args.result_dir, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_xlsx', type=str, required=True, help="filepath of reference xlsx")
    parser.add_argument('--result_dir', type=str, default='./result', help="directory containing clustering result json files")
    args = parser.parse_args()
    main(args)