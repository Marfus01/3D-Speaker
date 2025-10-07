#!/bin/bash
module purge
module load miniforge/24.1.2 cuda/11.8 gcc/9.3
source activate 3D-Speaker

data_root="/data/home/scv7387/run/tv_series_plus/dataset"
tv_name="the big bang theory" # "the big bang theory", "I love my family"
language="en" # 语言类型，支持 "en" 和 "zh-cn"
master_port=29667

bash /data/home/scv7387/run/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-diarization/run_video.sh --stage 1 --tv_name "$tv_name" --language $language --data_root "$data_root" --master_port $master_port