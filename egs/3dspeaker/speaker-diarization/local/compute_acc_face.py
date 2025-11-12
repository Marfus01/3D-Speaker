import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from acc_utils import *

def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    json_files = [os.path.join(args.result_dir, f) for f in os.listdir(args.result_dir) if f.endswith('.json') and os.path.isfile(os.path.join(args.result_dir, f))]  # all cluster json files for evaluation
    json_files = [f for f in json_files if 'faces' in os.path.basename(f) and 'mid_frame' in os.path.basename(f)]  # only evaluate face clustering results in mid-frame
    if not json_files:
        print("No face cluster json files found in", args.result_dir)
        sys.exit(1)
    
    for json_file in json_files:
        # 1. 读取聚类结果
        with open(json_file, 'r', encoding='utf-8') as f:
            cluster_dic = json.load(f)
        # 2. 读取标注文件，并筛选有标注的数据，提取对应的segment id，说话人和时长
        df = pd.read_excel(args.ref_xlsx) # Text Index列从 0 开始
        keys = df.apply(lambda row: f"{row['audio_seg_id']}_{int(row['face index'])}", axis=1)  # 与聚类结果中的 face ID 完全对应
        face_labels = df['face label'].tolist()
        face_others_set = set([face_label for face_label in face_labels if face_label not in main_character_list])
        print("Non-main character labels in the reference xlsx:", face_others_set)
        face_labels = ['Others' if face_label not in main_character_list else face_label for face_label in face_labels] # Replace all non-main characters with 'Others'
        face_sizes = df.apply(lambda row: (row['x2'] - row['x1']) * (row['y2'] - row['y1']), axis=1)

        # 3. 获取所有有标注数据的聚类标签
        cluster_labels = [cluster_dic.get(k, -2) for k in keys]
        valid_idx = [i for i, c in enumerate(cluster_labels) if c != -2]
        if len(valid_idx) < len(keys):
          missing_keys = [keys[i] for i in range(len(keys)) if i not in valid_idx]
          missing_keys_num = len(missing_keys)
          total_keys_num = len(keys)
          print(f"Warning: {missing_keys_num} out of {total_keys_num} keys in the reference xlsx are missing in {json_file}.")
          keys = [keys[i] for i in valid_idx]
          face_labels = [face_labels[i] for i in valid_idx]
          cluster_labels = [cluster_labels[i] for i in valid_idx]
          face_sizes = [face_sizes[i] for i in valid_idx]

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

        # 5. 根据标注文件，构建标注演员名->face id的字典
        unique_names = sorted(set(face_labels + ['Others']))
        name2idx = {name: idx for idx, name in enumerate(unique_names)}
        print("Character name to index mapping:", name2idx)

        # 6. 将 prediction 和 label 都转为 one-hot，随后利用匈牙利算法进行标签对齐
        face_labels_onehot = np.array([name2onehot(name, name2idx) for name in face_labels])
        n_clusters = max(cluster_labels) + 1
        if flag_has_neg1:
            n_clusters -= 1  # 不考虑 -1 对应的簇
            neg1_new_cluster_label = cluster_label_mapping[-1]
            if neg1_new_cluster_label == 0:
                mapping = {0: name2idx['Others']}  # 所有cluster id均为 -1 的特殊情况
            else:
                # 从聚类结果覆盖样本与标注数据覆盖样本的交集中，筛选聚类标签非 -1 的部分数据，用于构建聚类簇-->角色映射
                cluster_labels_filtered = [label for label in cluster_labels if label != neg1_new_cluster_label]
                faces_labels_filtered = [name2idx[face_labels[i]] for i in range(len(cluster_labels)) if cluster_labels[i] != neg1_new_cluster_label]
                # 用于构建cluster->character映射的部分标注不一定包含所有演员，需要重新编号
                face_label_mapping_temp = {name_idx: temp_idx for temp_idx, name_idx in enumerate(sorted(np.unique(faces_labels_filtered).tolist()))}
                face_label_mapping_temp_rev = {v: k for k, v in face_label_mapping_temp.items()}
                # 构建 one-hot 编码
                cluster_onehot_filtered = np.array(list(map(lambda x: list2onehot(x, n_clusters), cluster_labels_filtered)))
                faces_onehot_filtered =  np.array(list(map(lambda x: list2onehot(face_label_mapping_temp[x], len(face_label_mapping_temp)), faces_labels_filtered)))
                # 进行匹配
                mapping = class_matching(faces_onehot_filtered, cluster_onehot_filtered, others_chara_id=name2idx['Others'])
                mapping = {k: face_label_mapping_temp_rev[v] if v in face_label_mapping_temp_rev else name2idx['Others'] for k, v in mapping.items()} # 需要考虑筛选出的数据不包含others，但是class_matching由于n_class_ref<n_class_pred会自动将部分簇映射到others的情况
                mapping[neg1_new_cluster_label] = name2idx['Others']  # 将 -1 对应的簇映射到 'Others'
        else:
            cluster_labels_onehot = np.array(list(map(lambda x: list2onehot(x, n_clusters), cluster_labels)))
            mapping = class_matching(face_labels_onehot, cluster_labels_onehot, others_chara_id=name2idx['Others'])

        cluster_pred = np.eye(len(name2idx))[np.array([mapping[label] for label in cluster_labels])]
        print("Cluster_id to character_name_id mapping:", mapping)

        # 7. 按时长分组，计算分组/整体的 accuracy
        bins = [0, 10000, 20000, 30000, 40000, float('inf')]
        group_indices = np.digitize(face_sizes, bins) - 1  # 取值范围 [0, 4]，len(group_indices) == len(durations)
        results = {}
        results['overall_accuracy'] = cal_accuracy_onehot(face_labels_onehot, cluster_pred)
        for i in range(5):
            idx = [j for j, g in enumerate(group_indices) if g == i]
            if idx:
                acc = cal_accuracy_onehot(face_labels_onehot[idx], cluster_pred[idx])
                results[f'group_{i}_accuracy'] = acc
        # 7.2 按真实人脸计算 accuracy
        name2idx_sorted = sorted(name2idx.items(), key=lambda x:  face_labels_onehot[:, x[1]].sum(), reverse=True)  # 按说话人出现次数排序
        for name, idx in name2idx_sorted:
            idxs = [i for i in range(len(face_labels_onehot)) if face_labels_onehot[i][idx] == 1]
            if idxs:
                acc = cal_accuracy_onehot(face_labels_onehot[idxs], cluster_pred[idxs])
                results[f'accuracy_{name}'] = acc

        # 8. 保存结果
        filename = os.path.basename(json_file).replace('.json', '_accuracy.txt')
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
    args = parser.parse_args()
    main(args)