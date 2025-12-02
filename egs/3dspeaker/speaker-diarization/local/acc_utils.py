import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

main_character_list_IL = ['傅老', '和平', '志新', '志国', '圆圆', '小凡', '小张', '燕红']
main_character_list_BB = ['Sheldon', 'Leonard', 'Penny', 'Howard', 'Raj']
main_character_list = main_character_list_IL + main_character_list_BB

def eval_test_split(xlsx_path):
   pass

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
        pad_val = np.min(count_matrix) - 1
        padded_matrix = np.pad(count_matrix, ((0,0),(0,n_ref-n_pred)), constant_values=pad_val)
        row_ind, col_ind = linear_sum_assignment(-padded_matrix)
        valid = col_ind < n_pred
        row_ind = row_ind[valid]
        col_ind = col_ind[valid]
    else:
        assert others_chara_id is not None, "When n_ref < n_pred, others_chara_id must be provided."
        # 行补齐
        pad_val = np.min(count_matrix) - 1
        padded_matrix = np.pad(count_matrix, ((0,n_pred-n_ref),(0,0)), constant_values=pad_val)
        row_ind, col_ind = linear_sum_assignment(-padded_matrix)
        valid = row_ind < n_ref
        row_ind_valid = row_ind[valid]        
        col_ind_valid = col_ind[valid]
        col_ind_others = col_ind[~valid]
        if len(col_ind_others) > 0:
            row_ind_others = np.array([others_chara_id]*len(col_ind_others))
            row_ind = np.concatenate((row_ind_valid, row_ind_others))
            col_ind = np.concatenate((col_ind_valid, col_ind_others))
        else:
            row_ind = row_ind_valid
            col_ind = col_ind_valid        

    # 构建映射字典：pred_class -> ref_class。考虑了n_ref < n_pred时，label_ref中不存在others，但是部分簇需要映射到others的情况
    mapping = {pred_classes[col]: ref_classes[row] if row in ref_classes else others_chara_id for row, col in zip(row_ind, col_ind)}
    return mapping


# calculate accuracy given one-hot encoded label and prediction for multi-class classification
def cal_accuracy_onehot(label_onehot, pred_onehot):
    assert label_onehot.shape == pred_onehot.shape, "Shape of label and prediction must be the same."
    n_samples = label_onehot.shape[0]
    correct = np.sum(np.all(label_onehot == pred_onehot, axis=1))
    accuracy = correct / n_samples if n_samples > 0 else 0.0
    return round(accuracy, 4)
