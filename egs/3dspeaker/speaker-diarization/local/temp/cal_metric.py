import numpy as np
import pandas as pd
import soundfile as sf
import os, re, glob, sys
from scipy.optimize import linear_sum_assignment
sys.path.append("D:/wangchen/Research/tv_series/code") 
from src.utils import *

# correct order
characters_index_dic_BB = {'Sheldon': 0, 'Leonard': 1, 'Penny': 2, 'Howard': 3, 'Raj': 4, 'Others': 5}
characters_index_dic_IL = {'傅老': 0, '和平': 1, '志新': 2, '志国': 3, '圆圆': 4, '小凡': 5, '小张': 6, '燕红': 7, 'Others': 8}

TV_name = 'the big bang theory' # 'the big bang theory' #'I love my family'
def get_audio_len(TV_name, key):
    key_new = re.sub(r'^E(\d{1}-)', r'E0\1', key)
    audio_file = f'F:\\data\\TV_series\\tv_data\\{TV_name}\\triples_clean\\speaker_audio\\{key_new}.wav'
    data, samplerate = sf.read(audio_file)
    return len(data) / samplerate

# get episode and line_index of each row
def get_episode_line_index(TV_name):
    txt_path = f"F:\\data\\TV_series\\tv_data\\{TV_name}\\test_triple\\text_annotate_all.txt"
    ds_audio_path = os.path.join('F:\\data\\TV_series\\tv_data', TV_name, 'triples_clean', 'speaker_audio')

    # get the list of basename for audio files
    audio_list = glob.glob(os.path.join(ds_audio_path, '*.wav'))  # all audio files for training, sorted by name
    audio_list.sort(key=lambda x: tuple(map(int, re.match(r'E0*(\d+)-(\d+)', os.path.basename(x)).groups())))
    audio_list = list(map(lambda x: os.path.basename(x), audio_list))

    # read the text file, select the lines matching the audio files
    with open(txt_path, 'r', encoding='utf-8') as file:
      text = file.readlines()
    file.close()
    episode_line_index = [(line.split('|')[0], line.split('|')[1]) for line in text]
    episode_line_index = list(filter(lambda x: f'E{x[0]}-{x[1]}.wav' in audio_list, episode_line_index))
    # create a dictionary for episode and line index
    episode_line_index_dict = {f'E{ep.lstrip("0")}-{line}': idx for idx, (ep, line) in enumerate(episode_line_index)}
    return episode_line_index_dict


def cal_accuracy(speaker_true, speaker_preds, test_keys_list, true_index_dic, pred_index_dic):
    test_ind_xls = list(map(true_index_dic.get, test_keys_list)) 
    test_ind_preds = list(map(pred_index_dic.get, test_keys_list))
    accuracy, _ = cal_precision_recall(speaker_true[test_ind_xls], speaker_preds[test_ind_preds])
    return round(accuracy, 4)

def summary_accuracy(speaker_true, speaker_preds, test_keys_group_list, true_index_dic, pred_index_dic, remap=False):
  # test_keys_group_list: list of list of audio names
  test_sample_all = [item for sublist in test_keys_group_list for item in sublist]
  test_sample_all.sort(key=lambda x: tuple(map(int, re.match(r'E(\d+)-(\d+)', x).groups())))
  if remap: # adjust the order of columns in the speaker_pred to match the speaker_true
    map_init = class_matching(speaker_true, speaker_preds[list(map(pred_index_dic.get, test_sample_all))])
    speaker_preds_list = np.array([map_init[label] for label in np.argmax(speaker_preds, axis=1)])
    speaker_preds_aligned = np.eye(speaker_true.shape[1])[speaker_preds_list]
  else:
    speaker_preds_aligned = speaker_preds
  general_accuracy = cal_accuracy(speaker_true, speaker_preds_aligned, test_sample_all, true_index_dic, pred_index_dic)
  accuracy_group = list(map(lambda x: cal_accuracy(speaker_true, speaker_preds_aligned, x, true_index_dic, pred_index_dic), test_keys_group_list))
  accuracy_dic = {'general_accuracy': general_accuracy}
  for i, group in enumerate(accuracy_group):
    accuracy_dic[f'accuracyforgroup_{i}'] = group
  return accuracy_dic

def class_matching(onehot_ref, onehot_cur):
    label_ref = np.argmax(onehot_ref, axis=1)
    label_cur = np.argmax(onehot_cur, axis=1)
    p = len(np.unique(label_ref))
    count_matrix = np.zeros((p, p), dtype=int)

    for i in range(len(label_ref)):
        count_matrix[label_ref[i], label_cur[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-count_matrix)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    return mapping

def main():
    if TV_name == 'I love my family':
        characters_index_dic = characters_index_dic_IL
    elif TV_name == 'the big bang theory':
        characters_index_dic = characters_index_dic_BB
    else:
        raise ValueError('TV name not recognized')
    
    xls_file = f'F:\\data\\TV_series\\tv_data\\{TV_name}\\text_annotated.xlsx'
    xls_df = pd.read_excel(xls_file, sheet_name='Sheet1')
    speaker_true = np.array(xls_df['speaker'].apply(lambda x: name2binaries(x, characters_index_dic)).tolist())
    
    # print unkonwn speakers
    speaker = xls_df['speaker'].unique().tolist()
    unknown_speaker = [name for name in speaker if name not in characters_index_dic.keys()]
    print('Unknown speakers:', unknown_speaker)


    # choose useful rows in predictions for evaluation
    episode_line_index_dict = get_episode_line_index(TV_name)
    keys = xls_df.apply(lambda row: f"E{row['Episode']}-{row['Text Index']}", axis=1)
    df_index_dic = {key: idx for idx, key in enumerate(keys)}
    ## divide keys into 5 groups
    keys_len_dic = dict(map(lambda key: (key, get_audio_len(TV_name, key)), keys))  # get length_dic of audios
    bins = [0, 1, 2, 3, 4, float('inf')]
    key_groups = {i: [] for i in range(5)}
    for key, length in keys_len_dic.items():
      group = np.digitize(length, bins) - 1
      key_groups[group].append(key)
    key_groups = list(key_groups.values())
    print('Group size:', [len(group) for group in key_groups])
    print('Size of testset:', sum([len(group) for group in key_groups]))
    
    # exp_types = ['pretrain', 'contrastive','pretrain-hmm', 'contrastive-hmm']
    # exp_types = ['contrastive-hmm_v2_fh1', 'contrastive-hmm_v2_fh2', 'contrastive-hmm_v2_fh3'] # f use initial face cluster, fh use face prediction of face_HMM
    # exp_types = ['contrastive-baseline_new', 'contrastive-baseline_new2','contrastive-baseline_new3', 'contrastive-baseline_new4', 'contrastive-baseline_new5', 'contrastive-baseline_new6', 'contrastive-baseline_new7', 'contrastive-baseline_new8'] # 'baseline' is a naive way defined by myself, 'contrastive-baseline' used clustering-ensemble method
    exp_types = ['contrastive-baseline', 'contrastive-hmm_v2_fh_best']
    print(f'For {TV_name}:')
    for exp_type in exp_types:
        if 'v2_' in exp_type:
            data_folder = f'D:\\wangchen\\Research\\tv_series\\code\\runs\\{TV_name}\\train\\{exp_type}\\result'
        else:
            data_folder = f'D:\\wangchen\\Research\\tv_series\\code\\runs\\{TV_name}\\eval\\{exp_type}\\eval_results'
        if os.path.exists(data_folder) == False:
            continue
        else:
            print('') 
        
        if 'contrastive-baseline' in exp_type:                      
            if not os.path.exists(os.path.join(data_folder, 'mappings.npy')):    
                raise ValueError('No mappings.npy found!') 
            maps = np.load(os.path.join(data_folder, 'mappings.npy'), allow_pickle=True).item()
            map_init, map_best, map_final = maps['map_init'], maps['map_best'], maps['map_final']            
            speaker_pred_init = np.load(f'{data_folder}\\speaker_pred_init.npy')
            speaker_pred_best = np.load(f'{data_folder}\\speaker_pred.npy')
            # speaker_pred_final = np.load(f'{data_folder}\\speaker_pred_last.npy')
            speaker_pred_init = np.eye(speaker_true.shape[1])[np.array([map_init[label] for label in np.argmax(speaker_pred_init, axis=1)])]
            speaker_pred_best = np.eye(speaker_true.shape[1])[np.array([map_best[label] for label in np.argmax(speaker_pred_best, axis=1)])]
            # speaker_pred_final = np.eye(speaker_true.shape[1])[np.array([map_final[label] for label in np.argmax(speaker_pred_final, axis=1)])]

            # calculate the accuracys
            accuracy_dic_init = summary_accuracy(speaker_true, speaker_pred_init, key_groups, df_index_dic, episode_line_index_dict)
            accuracy_dic_best = summary_accuracy(speaker_true, speaker_pred_best, key_groups, df_index_dic, episode_line_index_dict)
            # accuracy_dic_final = summary_accuracy(speaker_true, speaker_pred_final, key_groups, df_index_dic, episode_line_index_dict)
            print(f'For {TV_name} with {exp_type} model')
            print(f'Initial speaker: {accuracy_dic_init}')
            print(f'Predicted speaker: {accuracy_dic_best}')
            # print(f'Final speaker: {accuracy_dic_final}')
        elif 'v2_' in exp_type:
            speaker_preds = np.load(os.path.join(data_folder, 'speaker_preds.npy'), allow_pickle=True).item()
            speaker_obs_0, speaker_hmm_0, = speaker_preds['speaker_obs_0'], speaker_preds['speaker_hmm_0']
            speaker_obs_1, speaker_hmm_1 = speaker_preds['speaker_obs_1'], speaker_preds['speaker_hmm_1']
            speaker_obs_l, speaker_hmm_l = speaker_preds['speaker_obs_l'], speaker_preds['speaker_hmm_l']
            # calculate the accuracys
            accuracy_dic_obs_0 = summary_accuracy(speaker_true, speaker_obs_0, key_groups, df_index_dic, episode_line_index_dict)
            accuracy_dic_hmm_0 = summary_accuracy(speaker_true, speaker_hmm_0, key_groups, df_index_dic, episode_line_index_dict)
            accuracy_dic_obs_1 = summary_accuracy(speaker_true, speaker_obs_1, key_groups, df_index_dic, episode_line_index_dict)
            accuracy_dic_hmm_1 = summary_accuracy(speaker_true, speaker_hmm_1, key_groups, df_index_dic, episode_line_index_dict)
            accuracy_dic_obs_l = summary_accuracy(speaker_true, speaker_obs_l, key_groups, df_index_dic, episode_line_index_dict)
            accuracy_dic_hmm_l = summary_accuracy(speaker_true, speaker_hmm_l, key_groups, df_index_dic, episode_line_index_dict)
            print(f'For {TV_name} with {exp_type} model')
            print(f'Observed speaker_0: {accuracy_dic_obs_0}')
            print(f'HMM speaker_0: {accuracy_dic_hmm_0}')
            print(f'Observed speaker_1: {accuracy_dic_obs_1}')
            print(f'HMM speaker_1: {accuracy_dic_hmm_1}')
            print(f'Observed speaker_l: {accuracy_dic_obs_l}')
            print(f'HMM speaker_l: {accuracy_dic_hmm_l}')
        
        else:
            speaker_obs = np.load(f'{data_folder}\\speaker_label_old.npy')
            pred_speaker_hmm = np.load(f'{data_folder}\\speaker_label.npy')
            # calculate the accuracys
            accuracy_dic_obs = summary_accuracy(speaker_true, speaker_obs, key_groups, df_index_dic, episode_line_index_dict)
            accuracy_dic_hmm = summary_accuracy(speaker_true, pred_speaker_hmm, key_groups, df_index_dic, episode_line_index_dict)
            print(f'For {TV_name} with {exp_type} model')
            print(f'Observed speaker: {accuracy_dic_obs}')
            print(f'HMM speaker: {accuracy_dic_hmm}')

        if exp_type == 'contrastive-hmm':
            test_sample_all = [item for sublist in key_groups for item in sublist]
            test_ind_xls = list(map(df_index_dic.get, test_sample_all))
            test_ind_preds = list(map(episode_line_index_dict.get, test_sample_all))
            np.save(f'D:\\wangchen\\Research\\tv_series\\code\\visual\\data\\rda\\cm\\{TV_name}\\speaker\\reference\\speaker_true.npy', speaker_true[test_ind_xls])
            np.save(f'D:\\wangchen\\Research\\tv_series\\code\\visual\\data\\rda\\cm\\{TV_name}\\speaker\\reference\\speaker_obs_selected.npy', speaker_obs[test_ind_preds])


if __name__ == '__main__':
    main()