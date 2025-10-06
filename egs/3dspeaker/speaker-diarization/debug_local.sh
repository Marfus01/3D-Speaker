#!/bin/bash
source activate 3D-Speaker

data_root=/f/data/tv_series_plus/tv_data
tv_name="debug_local" # "the big bang theory", "I love my family"
language="en" # 语言类型，支持 "en" 和 "zh-cn"

bash run_video.sh --stage 3 --tv_name $tv_name --language $language --data_root $data_root