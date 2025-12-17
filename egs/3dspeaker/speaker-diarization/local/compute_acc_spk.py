import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from acc_utils import *

def main(args):
    assert os.path.isfile(args.ref_xlsx), f"Reference xlsx file {args.ref_xlsx} does not exist."
    valid_keys_path = os.path.join(os.path.dirname(args.ref_xlsx), 'valid_part_keys_speaker.npy')
    if os.path.isfile(valid_keys_path):
        valid_keys_list = np.load(valid_keys_path, allow_pickle=True).tolist()
    else:
        # 拆分 valid/test 集合
        valid_keys_list = eval_test_split(args.ref_xlsx)
        np.save(valid_keys_path, np.array(valid_keys_list))
        print(f"Saved {len(valid_keys_list)} valid keys to {valid_keys_path}")

    os.makedirs(args.result_dir, exist_ok=True)
    json_files = [os.path.join(args.result_dir, f) for f in os.listdir(args.result_dir) if f.endswith('.json') and os.path.isfile(os.path.join(args.result_dir, f))]  # all cluster json files for evaluation
    json_files = [f for f in json_files if 'faces' not in os.path.basename(f)]  # only evaluate speaker clustering results based on audio or audio+vision
    if not json_files:
        print("No speaker cluster json files found in", args.result_dir)
        sys.exit(1)
    
    for json_file in json_files:
        # 1. 读取聚类结果
        with open(json_file, 'r', encoding='utf-8') as f:
            cluster_dic = json.load(f)
        # 2. 读取标注文件，并筛选有标注的数据，提取对应的segment id，说话人和时长
        df = pd.read_excel(args.ref_xlsx) # Text Index列从 0 开始
        df = df[df['whether annotate speaker'] == 'Yes']
        keys = df.apply(lambda row: f"E{int(row['Episode']):02}-{int(row['Text Index'])}", axis=1)  # 与聚类结果中的 segment ID 完全对应
        ## 根据 mode 筛选数据
        if args.mode != 'all':
            if args.mode == 'valid':
                df = df[keys.isin(valid_keys_list)]  # 筛选 keys 在 valid_keys_list 中的行
            elif args.mode == 'test':
                df = df[~keys.isin(valid_keys_list)]  # 筛选 keys 不在 valid_keys_list 中的行
            else:
                raise ValueError("Invalid mode. Choose from 'valid' or 'test'.")
            keys = df.apply(lambda row: f"E{int(row['Episode']):02}-{int(row['Text Index'])}", axis=1)  # 重新提取 keys
        ## 提取对应的segment id，说话人和时长
        speaker_labels = df['speaker'].tolist()
        speaker_others_set = set([speaker for speaker in speaker_labels if speaker not in main_character_list])
        print("Non-main character labels in the reference xlsx:", speaker_others_set)
        speaker_labels = ['Others' if speaker not in main_character_list else speaker for speaker in speaker_labels] # Replace all non-main characters with 'Others'
        print(f"Count of speaker labels in the reference xlsx ({args.mode} set):")
        print(Counter(speaker_labels))
        durations = df.apply(lambda row: time_to_seconds(row['End Time']) - time_to_seconds(row['Start Time']), axis=1)

        # 3. 获取所有有标注数据的聚类标签
        cluster_labels = [cluster_dic.get(k, -2) for k in keys]
        valid_idx = [i for i, c in enumerate(cluster_labels) if c != -2]
        if len(valid_idx) < len(keys):
          missing_keys = [keys.iloc[i] for i in range(len(keys)) if i not in valid_idx]
          missing_keys_num = len(missing_keys)
          total_keys_num = len(keys)
          print(f"Warning: {missing_keys_num} out of {total_keys_num} keys in the reference xlsx are missing in {json_file}.")
          keys = [keys.iloc[i] for i in valid_idx]
          speaker_labels = [speaker_labels[i] for i in valid_idx]
          cluster_labels = [cluster_labels[i] for i in valid_idx]
          durations = [durations.iloc[i] for i in valid_idx]

        # 4. 调整 cluster_labels，使其从 0 开始连续编号
        print('Original cluster ids and their counts on labeled data:', {label: cluster_labels.count(label) for label in set(cluster_labels)})
        ## Create a mapping from unique cluster labels to consecutive integers
        unique_cluster_labels = sorted(set(cluster_labels)) # 将 unique cluster labels 从小到大排序
        flag_has_neg1 = -1 in unique_cluster_labels
        if flag_has_neg1: # 将 -1 映射到最后一个整数
            unique_cluster_labels.remove(-1)
            cluster_label_mapping = {label: idx for idx, label in enumerate(unique_cluster_labels)}
            cluster_label_mapping[-1] = max(unique_cluster_labels) + 1 if unique_cluster_labels else 0
        else:
            cluster_label_mapping = {label: idx for idx, label in enumerate(unique_cluster_labels)}
        ## Map cluster_labels to consecutive integers
        cluster_labels = [cluster_label_mapping[label] for label in cluster_labels]
        print('Renamed cluster ids and their counts on labeled data:', {label: cluster_labels.count(label) for label in set(cluster_labels)})

        # 5. 根据标注文件，构建说话人名->speaker id的字典
        unique_speakers = sorted(set(speaker_labels + ['Others']))
        name2idx = {name: idx for idx, name in enumerate(unique_speakers)}
        print("Speaker to index mapping:", name2idx)

        # 6. 将 prediction 和 label 都转为 one-hot，随后利用匈牙利算法进行标签对齐
        speaker_onehot = np.array([name2onehot(name, name2idx) for name in speaker_labels])
        n_clusters = max(cluster_labels) + 1
        if flag_has_neg1:
            n_clusters -= 1  # 不考虑 -1 对应的簇
            neg1_new_cluster_label = cluster_label_mapping[-1]
            if neg1_new_cluster_label == 0:
                mapping = {0: name2idx['Others']}  # 所有cluster id均为 -1 的特殊情况
            else:
                # 从聚类结果覆盖样本与标注数据覆盖样本的交集中，筛选聚类标签非 -1 的部分数据，用于构建聚类簇-->角色映射
                cluster_labels_filtered = [label for label in cluster_labels if label != neg1_new_cluster_label]
                speaker_labels_filtered = [name2idx[speaker_labels[i]] for i in range(len(cluster_labels)) if cluster_labels[i] != neg1_new_cluster_label]
                # 用于构建cluster->character映射的部分标注不一定包含所有演员，需要重新编号
                speaker_label_mapping_temp =  {name_idx: temp_idx for temp_idx, name_idx in enumerate(sorted(np.unique(speaker_labels_filtered).tolist()))}
                speaker_label_mapping_temp_rev = {v: k for k, v in speaker_label_mapping_temp.items()}
                # 构建 one-hot 编码
                cluster_onehot_filtered = np.array(list(map(lambda x: list2onehot(x, n_clusters), cluster_labels_filtered)))
                speaker_onehot_filtered = np.array(list(map(lambda x: list2onehot(speaker_label_mapping_temp[x], len(speaker_label_mapping_temp)), speaker_labels_filtered)))
                # 进行匹配
                mapping = class_matching(speaker_onehot_filtered, cluster_onehot_filtered, others_chara_id=name2idx['Others'])
                mapping = {k: speaker_label_mapping_temp_rev[v] if v in speaker_label_mapping_temp_rev else name2idx['Others'] for k, v in mapping.items()} # 需要考虑筛选出的数据不包含others，但是class_matching由于n_class_ref<n_class_pred会自动将部分簇映射到others的情况
                mapping[neg1_new_cluster_label] = name2idx['Others']  # 将 -1 对应的簇映射到 'Others'
        else:
            cluster_onehot = np.array(list(map(lambda x: list2onehot(x, n_clusters), cluster_labels)))
            mapping = class_matching(speaker_onehot, cluster_onehot, others_chara_id=name2idx['Others'])

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
        # 7.2 按真实说话人计算 accuracy
        name2idx_sorted = sorted(name2idx.items(), key=lambda x:  speaker_onehot[:, x[1]].sum(), reverse=True)  # 按说话人出现次数排序
        for name, idx in name2idx_sorted:
            idxs = [i for i in range(len(speaker_onehot)) if speaker_onehot[i][idx] == 1]
            if idxs:
                acc = cal_accuracy_onehot(speaker_onehot[idxs], cluster_pred[idxs])
                results[f'accuracy_{name}'] = acc


        # 8. 保存结果
        if args.mode == 'all':
            filename = os.path.basename(json_file).replace('.json', f'_accuracy.txt')
        else:
            filename = os.path.basename(json_file).replace('.json', f'_accuracy({args.mode}).txt')
        with open(os.path.join(args.result_dir, filename), 'w') as f:
            name_grp_cnt = 0
            for k, v in results.items():
                if k.startswith('accuracy_'):
                    if name_grp_cnt == 0:
                        f.write("\n")
                    name_grp_cnt += 1
                f.write(f"{k}: {v}\n")

        print("Accuracy results saved to", os.path.join(args.result_dir, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_xlsx', type=str, required=True, help="filepath of reference xlsx")
    parser.add_argument('--result_dir', type=str, default='./result', help="directory containing clustering result json files")
    parser.add_argument('--mode', type=str, default='test', help="mode: valid or test or all")
    args = parser.parse_args()
    main(args)