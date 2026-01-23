import numpy as np
import pandas as pd
import os, pickle, sys
from operator import itemgetter
from sklearn import manifold, metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

current_file_path = os.path.abspath(__file__)
# 从'visualization'回到'local/'目录
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path),'..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils import save_testEER, evaluate_EER

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