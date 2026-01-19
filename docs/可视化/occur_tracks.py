import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
import os, sys, re, json, pickle
# Add parent directory to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from egs.3dspeaker.speaker-diarization.local.acc_utils import get_map


name_dic_BB = {'Leonard': [['莱纳德·霍夫斯塔德博士'], 
                           ['霍夫斯塔德博士', '莱纳德·霍夫斯塔德'], 
                           ['莱纳德', '霍夫斯塔德']], 
               'Sheldon': [['谢尔顿·库珀博士'], 
                           ['谢尔顿·库珀', '库珀博士'], 
                           ['库珀', '谢尔顿']], 
               'Penny': [['佩妮']], 
               'Howard': [['霍华德·沃尔维茨', '霍华德·沃罗威茨'], 
                          ['霍华德', '沃尔维茨', '沃罗威茨']], 
               'Raj': [['拉杰·库萨帕里'], 
                       ['拉杰', '库萨帕里']], 
               'Others': [['小拉丽塔', '罗德·泰勒', '盖博豪斯博士'], 
                          ['丹尼斯', '伊米尔·法尔曼法尔米安博士', '克里斯汀', '克里顿', '凯尔', '卢克', '史巴克', '哈里', '哈雷', '埃尔文', '夏纳', '奥本海默', '威廉', '尼奥', '帕特尔', '帕萨迪纳', '弗拉多', '彼得', '德法布', '柯克舰长', '沃森博士', '沃罗威茨', '海伦', '潘查理公主', '爱因斯坦', '牛顿', '玛丽', '种马博士', '米茜', '艾尔顿·约翰', '艾瑞克', '苏菲', '莫扎特', '莫洛克', '莱恩小姐', '莱斯利', '谢利', '贝尔夫人', '贝尔特', '里克', '里奥', '金先生', '阿基米德', '雅克·库斯托', '麦克',
                          '拉丽塔', '泰勒', '盖博豪斯', '罗德']]
                          }

name_dic_IL = {'Fulao': [['老傅同志'], 
                        ['傅明同志', '傅某人', '傅伯伯', '傅局长', '老局长', '老傅', '傅老', '贾敬贤']],
               'Heping': [['和平同志', '和平老大妈', '和平女侠'], 
                        ['和平']],
               'Zhixin': [['志新哥哥', '贾志新同志'], 
                        ['贾志新', '志新同志', '志新哥'], 
                        ['贾总', '贾炕', '志新']],
               'Zhiguo': [['贾志国同志', '贾志国先生'], 
                        ['贾志国', '志国同志'], 
                        ['志国哥哥', '志国', '贾先生', '贾总']],
               'Yuanyuan': [['贾圆圆同学', '贾圆圆同志', '贾圆圆歌友'], 
                        ['贾圆圆'], 
                        ['圆圆', '坚妮']],
               'Xiaofan': [['贾小凡同志', '贾小凡同学'], 
                        ['贾小凡', '小凡姐', '小凡妹妹', '小凡姑娘'], 
                        ['小凡']],
               'Xiaozhang': [['小张阿姨', '凤姑妹子', '凤姑妹妹', '张凤姑', '张姑奶奶'], 
                        ['小张', '凤姑', '小保姆']],
               'Yanhong': [['燕红妹妹', '燕红阿姨', '郑燕红同志'], 
                        ['郑燕红'], 
                        ['燕红']],
               'Others': [['丽达姑娘','宝财哥', '纪春生', '春花姐'], 
                          ['三毛', '丽丽', '余主任', '余大妈', '余小姐', '侯小姐', '刘宇航', '刘建军', '刘文彩', '刘胡兰', '刘颖', '叶之贤', '周厂长', '姓孟的', '孟德', '孟昭晖', '孟老师', '小余', '小刘', '小加', '小史姑娘', '小方', '小晴', '小秘', '小翠', '小许', '尤主编', '张先生', '张国荣', '张欣来', '彭老师', '文怡', '方小姐', '昆儿哥',  '朱总司令', '李秘', '杨晶晶', '毛主席', '王总', '王老师', '童安格', '老二', '老和同志', '老四', '老头儿', '老李', '老玉米', '老赵', '老郑', '胡大个儿', '费丽斯', '邱少云', '郑伯伯', '郑千里', '郑爷爷', '阿东', '阿姨', '阿巩', '阿庆', '阿敏', '阿昆', '阿欢', '阿玉', '阿英', '阿荣', '顾科长', '马小姐', '齐大姐', '齐大姨',
                          '丽达', '宝财', '春生', '春花']]
                          }

def find_name_in_texts(name_dic, txt_folder, save_path):
    """在文本文件中查找人物名字出现的位置。
    Args:
        name_dic: 包含人物别名映射的字典
        txt_folder: 包含文本文件的文件夹路径
        save_path: 结果JSON文件的保存路径
    
    Returns:
        dict: key为人物名称，value为包含该人物别名出现的音频片段ID的列表
    """
    txt_files = [f for f in os.listdir(txt_folder) if f.lower().endswith('.txt')]
    name_dic_all = {k: [name for sublist in name_list for name in sublist] for k, name_list in name_dic.items()}
    result = {k: [] for k in name_dic.keys()}

    for file in sorted(txt_files):
        episode_name = os.path.splitext(file)[0]
        with open(os.path.join(txt_folder, file), 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.rstrip('\n')
                if not line:
                    continue
                
                parts = line.split('|')
                line_idx = int(parts[1]) - 1
                audio_seg_id = f"{episode_name}-{line_idx}"
                
                text = parts[-1]
                for key, names_all in name_dic_all.items():
                    counts = list(map(lambda name: len(re.findall(re.escape(name), text)), names_all))
                    if sum(counts) > 0:
                        result[key].append(audio_seg_id)

    with open(save_path, 'w', encoding='utf-8') as outf:
        json.dump(result, outf, ensure_ascii=False, indent=2)
    return result


TV_name = "the big bang theory"


# Load useful variables and pseudo labels
pseudo_label_dir = f"/data/home/scv7387/run/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-diarization/runs/{TV_name}/exp_video/result/self_supervised/2. 仅微调dense layer，从hid_feat构建数据集/exp27.7/round9/pseudo_label"

## Load lengths_episode and audio_seg_ids from useful_var_dic.pkl
useful_var_path = os.path.join(pseudo_label_dir, 'useful_var_dic.pkl')
assert os.path.exists(useful_var_path), f"When from_preds is True, useful_var_dic.pkl must exist in {pseudo_label_dir}."
with open(useful_var_path, 'rb') as f:
    useful_var_dic = pickle.load(f)
lengths_episode = useful_var_dic['alengths']
audio_seg_ids = useful_var_dic['audio_seg_ids']

## Load pseudo labels for audio and face modalities
### audio
pseudo_label_files_audio = [f for f in os.listdir(pseudo_label_dir) if f.endswith('.json') and 'pseudo_labels_audio' in f]
assert len(pseudo_label_files_audio) == 1, "There should be exactly one pseudo_labels_audio file."
with open(os.path.join(pseudo_label_dir, pseudo_label_files_audio[0]), 'r') as f:
    pseudo_labels_audio = json.load(f)
### face
pseudo_label_files_face = [f for f in os.listdir(pseudo_label_dir) if f.endswith('.json') and 'pseudo_labels_faces_mid_frame_train' in f]
assert len(pseudo_label_files_face) == 1, "There should be exactly one pseudo_labels_faces_mid_frame_train file."
with open(os.path.join(pseudo_label_dir, pseudo_label_files_face[0]), 'r') as f:
    pseudo_labels_faces = json.load(f)

# Load name entity occurrence results
if TV_name == 'the big bang theory':
    name_dic = name_dic_BB
    txt_folder = f"/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/speaker_text_cn"
elif TV_name == 'I love my family':
    name_dic = name_dic_IL
    txt_folder = f"/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/speaker_text"
else:
    raise ValueError("TV_name should be 'the big bang theory' or 'I love my family'.")
name_occur_dict = find_name_in_texts(name_dic, txt_folder, save_path=f"/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/name2audio_ids_dic.json")


# align audio cluster ids with names
## get count_matrix of co-occurance of name entity and speaker cluster id
spk_cluster_ids_all = list(set(pseudo_labels_audio.values()))
spk_cluster_ids_to_match = sorted([cid for cid in spk_cluster_ids_all if cid >=0])  # exclude negative cluster ids
names_to_match = list(name_dic.keys())
spk2idxs_dic = {cluster_id: idx for idx, cluster_id in enumerate(spk_cluster_ids_to_match)}
name2idxs_dic = {name: idx for idx, name in enumerate(names_to_match)}
n_name, n_spk = len(names_to_match), len(spk_cluster_ids_to_match)
count_matrix = np.zeros((n_name, n_spk), dtype=int)  # rows: names, cols: speaker cluster ids
for name, audio_seg_id_list in name_occur_dict.items():
    name_idx = name2idxs_dic[name]
    for audio_seg_id in audio_seg_id_list:
        spk_cluster_id = pseudo_labels_audio[audio_seg_id]
        if spk_cluster_id < 0:
            continue
        spk_idx = spk2idxs_dic[spk_cluster_id]
        count_matrix[name_idx, spk_idx] += 1
others_chara_idx = name2idxs_dic['Others']
row_ind, col_ind = get_map(-count_matrix, others_chara_idx) # spk don't mention self, so use negative count_matrix
idx2spk_dic = {idx: cluster_id for cluster_id, idx in spk2idxs_dic.items()}
idx2names_dic = {idx: name for name, idx in name2idxs_dic.items()}
spk2name_map = {idx2spk_dic[col]: idx2names_dic[row] if row in idx2names_dic else 'Others' for row, col in zip(row_ind, col_ind)}



# print(len(face_label), len(name_flags), len(speaker_label))