import os, re
import numpy as np
import torch.nn as nn
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# get Get the last but (index-1) parent directory name of a movie file.
def get_parent_dir(file_path,index):  
    drive, path_in_disk = os.path.splitdrive(file_path)
    index_level_root = os.path.join(drive+os.sep, *path_in_disk.split(os.sep)[:-(index)])
    return index_level_root

# get the corresponding filename for each row in the DataFrame
def get_row_name(row):
  episode = row['Episode']
  text_index = row['Text Index']
  return(f'E{int(episode):02}-{int(text_index)}')



# given a list of feature's index, convert the index to the corresponding file name
def map_index2name(lst, dic):
  mapped_list = list(map(lambda x: dic[x], lst))
  mapped_list.sort(key=lambda x: tuple(map(int, re.match(r'E0*(\d+)-(\d+)', x).groups())))
  return mapped_list


# clustering for feat
def clustering_feats(feats, min_similarity):
    ### get the cosine similarity matrix
    feats_arr = feats # np.array of shape (n_samples, n_features)
    cosine_dis = cosine_similarity(feats_arr)
    edges_mat = (cosine_dis > min_similarity)
    del cosine_dis

    ### Construct a connected graph through depth-first search, and then get the connected components
    graph = nx.from_numpy_array(edges_mat)
    del edges_mat
    components = [list(c) for c in nx.connected_components(graph)]
    components.sort(key=len, reverse=True)
    del graph

    return components


# Given audio files of speaker class \rho'' and a dictionary of frame classes, calculate the similarity factor (SF) between the audio files and each frame class.
def cal_SF(audio_list, frame_list):
  return len(set(audio_list) & set(frame_list))/len(set(frame_list))

# map cluster of audio files to the corresponding frame class according to the similarity factor
def map_audio2face(audio_list, frame_class_dic):
  audio2face = {fame_class: cal_SF(audio_list, frame_list) for fame_class, frame_list in frame_class_dic.items()}
  max_key = max(audio2face, key=audio2face.get)
  return max_key


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

# given a face existing vector, convert it to probability distribution
def facevec2prob(face_vec, others_ratio):
  n_faces = len(face_vec)
  if np.sum(face_vec) == 0: # uniform distribution is no face detected
    prob = np.ones(n_faces) / n_faces
  else:
    prob = face_vec / np.sum(face_vec)

  if prob[-1]>0:  # if 'others' face detected, allocate some prob to main actors, otherwise others face are too many
    if np.random.uniform(0, 1) > others_ratio:
      prob[:-1] += prob[-1]*0.7/(n_faces-1)
      prob[-1] = 0.3*prob[-1]
      prob = prob/np.sum(prob)

  return prob # sum(prob) may not be 1 due tothe tolerance of float


# given a dic of audio name to class, convert it to a np.array of 0-1 encoding
def dic2seq(dic, n_states, line_index_list, max_Episode):
  obs = [[] for _ in range(max_Episode)]

  for i in range(max_Episode):  # the first [] can be removed for face, since the value is a list
    obs[i] = list(map(lambda x: list2onehot(dic[f'E{str(i+1).zfill(2)}-{str(x)}'], n_states), line_index_list[i]))

  lengths = list(map(len, obs))
  obs_arr = np.concatenate(obs)
  return obs_arr, lengths


# calculate TP, FP, FN
def cal_TP_FP_FN(label, pred):
  if label.shape != pred.shape:
    raise ValueError(f"Shape of label is {label.shape}, but shape of pred is {pred.shape}.")
  pred_flat = pred
  label_flat = label
  if pred_flat.ndim > 1:
    pred_flat = pred_flat.reshape(-1)
  if label_flat.ndim > 1:
    label_flat = label_flat.reshape(-1)
  TP = np.sum((pred_flat == 1) & (label_flat == 1))
  FP = np.sum((pred_flat == 1) & (label_flat == 0))
  FN = np.sum((pred_flat == 0) & (label_flat == 1))
  return TP, FP, FN

# calculate precision and recall
def cal_precision_recall(label, pred):
  TP, FP, FN = cal_TP_FP_FN(label, pred)
  precision = TP/(TP+FP) if TP+FP>0 else 0
  recall = TP/(TP+FN) if TP+FN>0 else 0
  return precision, recall



# Initialize the weight of Classifier in Trainer
def weights_init_mlp(m):
  if isinstance(m, nn.Linear):
    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    if m.bias is not None:
      nn.init.constant_(m.bias, 0)

# convert the name string to int
def name2binaries(name, characters_index_dic):
  if name == '' or name == 'nan':
    result = [0]*len(characters_index_dic)
  elif name in characters_index_dic:
    result = list(map(lambda x: 1 if x == characters_index_dic[name] else 0, range(len(characters_index_dic))))
  else:
    result = list(map(lambda x: 1 if x == characters_index_dic['Others'] else 0, range(len(characters_index_dic))))
  
  return result

# convert the label string to index list
def str2binaries(s, characters_index_dic):
  if s == '' or s == 'nan':
    result = [0]*len(characters_index_dic)
  else:
    name_list = list(map(lambda x: x.strip(), s.split(',')))
    result = list(map(lambda name: name2binaries(name, characters_index_dic), name_list))
    result = [1 if x > 0 else 0 for x in np.sum(np.array(result), axis=0).tolist()]
  return result