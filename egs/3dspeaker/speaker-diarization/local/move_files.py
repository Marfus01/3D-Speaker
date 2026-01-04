import json
import os, shutil

TV_name = 'I love my family' # 'the big bang theory', 'I love my family'
pseudo_labels_json = f'/data/home/scv7387/run/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-diarization/runs/{TV_name}/exp_video/result/self_supervised/initial_v1.1/pseudo_label/cluster_results_faces_mid_frame_processed_all_for_HMM_nested_X.json'
# Load pseudo-labels
with open(pseudo_labels_json, 'r') as f:
    pseudo_labels = json.load(f)
sample_ids_mf = list(pseudo_labels.keys())

# Create label encoder (map cluster IDs to continuous class indices)
unique_labels = sorted(list(set([pseudo_labels[sid] for sid in sample_ids_mf])))
# Map from label to list of sample IDs
label2samples_dict = {label: [] for label in unique_labels}
for sid in sample_ids_mf:
    label = pseudo_labels[sid]
    label2samples_dict[label].append(sid)

# move files according to label2samples_dict
dataset_dir = f'/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/midframe_faces'
output_dir = f'/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/midframe_faces_clustered'
os.makedirs(output_dir, exist_ok=True)
for label, sample_ids in label2samples_dict.items():
    label_dir = os.path.join(output_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)
    samples_path_src = list(map(lambda sid: os.path.join(dataset_dir, sid.rsplit('-', 1)[0], f'{sid}.jpg'), sample_ids))
    samples_path_dst = list(map(lambda sid: os.path.join(label_dir, f'{sid}.jpg'), sample_ids))
    _=list(map(lambda src, dst: shutil.copyfile(src, dst), samples_path_src, samples_path_dst))