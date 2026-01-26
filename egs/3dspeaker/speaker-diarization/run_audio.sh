#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script performs speaker diarization task based on audio-only input, 
# in contrast to "run_video.sh" which is based on video and audio input.

set -e
. ./path.sh || exit 1

stage=1
stop_stage=7

conf_file=conf/diar.yaml  # 在说话人特征提取时，仅作为template使用，实际参数均在脚本中指定
gpus="0"
nj=1  # 对应说话人嵌入提取和聚类时的threads_num。应当是gpus_num的整数倍
master_port=29567

language="en" # 语言类型，支持 "en" 和 "zh-cn"
# Set speaker_model_id to damo/speech_eres2net_sv_zh-cn_16k-common when using eres2net
if [ "$language" = "en" ]; then
  speaker_model_id=iic/speech_campplus_sv_en_voxceleb_16k
elif [ "$language" = "zh-cn" ]; then
  speaker_model_id=iic/speech_campplus_sv_zh-cn_3dspeaker_16k
else
  echo "Only support 'en' and 'zh-cn' for language now. Exit with error."
  exit 1
fi
from_subtitle=false  # 是否直接从字幕文件中提取说话人分割信息
include_overlap=true
hf_access_token=  # 用于访问 "pyannote/segmentation-3.0" 模型的 HuggingFace 访问令牌

# Contrastive learning parameters
contrastive_training_flag=false  # 是否在提取embedding之前进行对比学习
contrastive_lr=0.0001  # 对比学习学习率
contrastive_batch_size=128  # 对比学习batch size
contrastive_max_dur=2.0  # 对比学习最大持续时间（秒）
contrastive_max_epochs=100  # 对比学习最大epoch数
contrastive_test_interval=1  # 对比学习评估间隔
contrastive_early_stop_patience=10  # 对比学习早停patience
musan_path=""  # MUSAN噪声数据集路径
rir_filepath=""  # RIR噪声文件路径
testEER_file=""  # testEER文件路径（用于对比学习评估）

examples=examples
exp=exp

. local/parse_options.sh || exit 1

wav_list=$examples/wav.list
json_dir=$exp/json
embs_dir=$exp/embs
rttm_dir=$exp/rttm

if [ "$include_overlap" = true ] && [ -z "$hf_access_token" ]; then
  echo "[ERROR]: The hf_access_token is empty. If \"include_overlap\" is set to true,\
    the \"hf_access_token\" for \"pyannote/segmentation-3.0\" should be provided." && exit 1
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  if [ ! -f "$wav_list" ]; then
    echo "$(basename $0) Stage 1: Prepare input wavs..."
    mkdir -p $examples
    wget "https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common/\
resolve/master/examples/2speakers_example.wav" -O $examples/2speakers_example.wav
    wget "https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common/\
resolve/master/examples/2speakers_example.rttm" -O $examples/2speakers_example.rttm
    echo "examples/2speakers_example.wav" > $examples/wav.list
    echo "examples/2speakers_example.rttm" > $examples/refrttm.list
  else
    echo "$(basename $0) Stage 1: $wav_list exists. Skip this stage."
  fi
fi

###### Begin extracting speaker embeddings ######
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # 对于wav_list（list of unsegmented wav file_paths）包含的每个音频文件，使用pyannote/segmentation-3.0做重叠说话人检测，汇总为 dict 后保存为 pkl。默认不进行。
  if [ "$include_overlap" = true ]; then
    echo "$(basename $0) Stage2: Do overlap detection for input wavs..."
    python local/overlap_detection.py --wavs "$wav_list" --out_dir "$json_dir" --hf_access_token "$hf_access_token"
  fi
  if [ "$from_subtitle" = false ]; then
    echo "$(basename $0) Stage2: Do vad for input wavs..."
    # 对于wav_list（list of unsegmented wav file_paths）包含的每个音频文件，使用FSMN-Monophone VAD提取其中每段有效语音的起止时间点，汇总写入exp_video/json/vad.json
    python local/voice_activity_detection.py --wavs "$wav_list" --out_file "$json_dir/vad.json"
  fi
fi

# 使用滑动窗口(滑动步长 = 0.75, 窗宽 = 1.5)，将vad.json中记录的每段有效语音进一步切分为多个子片段，汇总写入exp_video/json/subseg.json
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  if [ "$from_subtitle" = false ]; then
    echo "$(basename $0) Stage3: Prepare subsegments info..."
    python local/prepare_subseg_json.py --vad "$json_dir/vad.json" --out_file "$json_dir/subseg.json"
    cp "$json_dir/subseg.json" "$json_dir/subseg_ori.json"
  else
    echo "$(basename $0) Stage3: Prepare audio&visual subsegments info from subtitle..."
    if [ -f "$json_dir/vad.json" ] && [ -f "$json_dir/subseg.json" ] && [ -f "$json_dir/subseg_ori.json" ]; then
      echo "$(basename $0) Stage3: $json_dir/vad.json, $json_dir/subseg.json and $json_dir/subseg_ori.json exist. Skip this stage."
    else
      python local/prepare_all_json_from_subtitle.py --wavs "$wav_list" \
      --out_file_vad "$json_dir/vad.json" --out_file_subseg "$json_dir/subseg.json" --out_file_subseg_ori "$json_dir/subseg_ori.json"
    fi
  fi
fi

# 对比学习训练（可选）: 在提取embedding之前，先对预训练模型在当前数据集上做对比学习
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && [ "$contrastive_training_flag" = true ]; then
  echo "$(basename $0) Stage3.5: Contrastive learning training..."
  
  # Check required parameters
  if [ -z "$musan_path" ] || [ -z "$rir_filepath" ]; then
    echo "Error: musan_path and rir_filepath are required for contrastive learning."
    exit 1
  fi
  
  if [ -z "$testEER_file" ]; then
    echo "Warning: testEER_file is not provided. Contrastive learning will skip evaluation."
    testEER_args=()
  else
    testEER_args=(--testEER_file "$testEER_file")
  fi
  
  # Create contrastive learning result directory
  contrastive_result_dir="$exp/contrastive_learning"
  mkdir -p "$contrastive_result_dir"
  # Copy conf_file to $exp/conf
  mkdir -p "$exp/conf"
  cp "$conf_file" "$exp/conf/"
  
  # Run contrastive learning training
  torchrun --nproc_per_node=$nj --master_port $master_port local/contrastive_finetune.py \
    --conf "$conf_file" \
    --subseg_json "$json_dir/subseg.json" \
    --musan_path "$musan_path" \
    --rir_filepath "$rir_filepath" \
    "${testEER_args[@]}" \
    --result_dir "$contrastive_result_dir" \
    --speaker_model_id "$speaker_model_id" \
    --lr "$contrastive_lr" \
    --batch_size "$contrastive_batch_size" \
    --max_dur "$contrastive_max_dur" \
    --max_epochs "$contrastive_max_epochs" \
    --test_interval "$contrastive_test_interval" \
    --early_stop_patience "$contrastive_early_stop_patience" \
    --gpu $gpus \
    --use_gpu \
    --seed 1234
  
  echo "$(basename $0) Stage3.5: Contrastive learning completed!"
  echo "Best model saved in $contrastive_result_dir/contrastive_models/"
fi

# 使用CAM++（中英文版）提取subseg.json中每个子片段的说话人嵌入，将每个原始音频文件的结果各自汇总为 dict 后，保存为exp_video/embs目录下同名的 pkl 文件。dict 的 key 是子片段的起止时间点(list)，value是说话人嵌入。
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "$(basename $0) Stage4: Extract speaker embeddings..."
  # Copy conf_file to $exp/conf
  mkdir -p "$exp/conf"
  cp "$conf_file" "$exp/conf/"
  
  # Update speaker_pretrained_model to use the best contrastive model
  speaker_pretrained_model_arg=()
  if [ "$contrastive_training_flag" = true ]; then
    echo "Contrastive learning training was performed in the previous stage."
    CONTRASTIVE_MODEL_PATH=$(find "$contrastive_result_dir/contrastive_models" -name "model_epoch_*.pth" -type f | sort -V | tail -1)
    if [ -f "$CONTRASTIVE_MODEL_PATH" ]; then
      echo "Using contrastive learning model: $CONTRASTIVE_MODEL_PATH"
      speaker_pretrained_model_arg=(--speaker_pretrained_model "$CONTRASTIVE_MODEL_PATH")
    fi
  fi

  # Extract speaker embeddings
  torchrun --nproc_per_node=$nj --master_port $master_port local/extract_diar_embeddings.py \
          --model_id $speaker_model_id "${speaker_pretrained_model_arg[@]}" --conf "$conf_file" \
          --subseg_ori_json "$json_dir/subseg_ori.json" --subseg_json "$json_dir/subseg.json" --embs_out "$embs_dir" --gpu $gpus --use_gpu
            
fi
###### End extracting speaker embeddings ######

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "$(basename $0) Stage5: Perform clustering and postprocessing, and output sys rttms..."
  if [ "$include_overlap" = true ]; then
    cluster_rttm_dir=$rttm_dir/intermediate
  else
    cluster_rttm_dir=$rttm_dir
  fi
  torchrun --nproc_per_node=$nj --master_port $master_port local/cluster_and_postprocess.py \
          --conf $conf_file --wavs $wav_list \
          --audio_embs_dir $embs_dir --rttm_dir $cluster_rttm_dir
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  if [ "$include_overlap" = true ]; then
    echo "$(basename $0) Stage6: Do overlap detection postprocess..."
    python local/refine_with_OD.py --init_rttm_dir $rttm_dir/intermediate --rttm_dir $rttm_dir \
        --segmentation_dir $json_dir --hf_access_token $hf_access_token
  fi
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "$(basename $0) Stage7: Get the DER metrics..."
  ref_rttm_list=$examples/refrttm.list
  if [ -f $ref_rttm_list ]; then
    cat $ref_rttm_list | while read line;do cat $line;done > $exp/concat_ref_rttm
    echo "Computing DER..."
    python local/compute_der.py --exp_dir $exp --ref_rttm $exp/concat_ref_rttm
  else
    echo "Refrttm.list is not detected. Can't calculate the result"
  fi
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "$(basename $0) Stage8: Generate segmented transcription results with speaker ID...This step may be time-consuming."
  torchrun --nproc_per_node=$nj --master_port $master_port local/out_transcription.py --exp_dir $exp --gpu $gpus
fi
