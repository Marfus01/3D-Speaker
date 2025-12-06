#!/bin/bash
module purge
module load miniforge/24.1.2 cuda/11.8 gcc/9.3
source activate 3D-Speaker

data_root="/data/home/scv7387/run/tv_series_plus/dataset"
tv_name="the big bang theory" # "the big bang theory", "I love my family"
language="en" # 语言类型，支持 "en" 和 "zh-cn"
unreliable_pp=5.0  # HMM平滑时，认为不可靠的说话人标签百分比，范围0-100.0
unfrozen_layers_num=13

bash /data/home/scv7387/run/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-diarization/run_video.sh --stage 5 --tv_name "$tv_name" --language $language --data_root "$data_root" --master_port 29697 --unreliable_pp $unreliable_pp --unfrozen_layers_num $unfrozen_layers_num