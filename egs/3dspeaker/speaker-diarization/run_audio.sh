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
language="en" # 语言类型，支持 "en" 和 "zh-cn"
from_subtitle=false  # 是否直接从字幕文件中提取说话人分割信息
include_overlap=false
hf_access_token=

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
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "$from_subtitle" = false ]; then
  # 对于wav_list（list of unsegmented wav file_paths）包含的每个音频文件，使用pyannote/segmentation-3.0做重叠说话人检测，汇总为 dict 后保存为 pkl。默认不进行。
  if [ "$include_overlap" = true ]; then
    echo "$(basename $0) Stage2: Do overlap detection for input wavs..."
    python local/overlap_detection.py --wavs $wav_list --out_dir $json_dir --hf_access_token $hf_access_token
  fi
  echo "$(basename $0) Stage2: Do vad for input wavs..."
  # 对于wav_list（list of unsegmented wav file_paths）包含的每个音频文件，使用FSMN-Monophone VAD提取其中每段有效语音的起止时间点，汇总写入exp_video/json/vad.json
  python local/voice_activity_detection.py --wavs $wav_list --out_file $json_dir/vad.json
fi

# 使用滑动窗口(滑动步长 = 0.75, 窗宽 = 1.5)，将vad.json中记录的每段有效语音进一步切分为多个子片段，汇总写入exp_video/json/subseg.json
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  if [ "$from_subtitle" = false ]; then
    echo "$(basename $0) Stage3: Prepare subsegments info..."
    python local/prepare_subseg_json.py --vad $json_dir/vad.json --out_file $json_dir/subseg.json
  else
    echo "$(basename $0) Stage3: Prepare audio&visual subsegments info from subtitle..."
    python local/prepare_all_json_from_subtitle.py --wavs "$wav_list" \
      --out_file_vad "$json_dir/vad.json" --out_file_subseg "$json_dir/subseg.json"
  
  fi
fi

# # 使用CAM++（中英文版）提取subseg.json中每个子片段的说话人嵌入，将每个原始音频文件的结果各自汇总为 dict 后，保存为exp_video/embs目录下同名的 pkl 文件。dict 的 key 是子片段的起止时间点(list)，value是说话人嵌入。
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#   echo "$(basename $0) Stage4: Extract speaker embeddings..."
#   # Set speaker_model_id to damo/speech_eres2net_sv_zh-cn_16k-common when using eres2net
#   if [ "$language" = "en" ]; then
#     speaker_model_id=iic/speech_campplus_sv_en_voxceleb_16k
#   elif [ "$language" = "zh-cn" ]; then
#     speaker_model_id=iic/speech_campplus_sv_zh-cn_3dspeaker_16k
#   else
#     echo "Only support 'en' and 'zh-cn' for language now. Exit with error."
#     exit 1
#   fi
#   torchrun --nproc_per_node=$nj local/extract_diar_embeddings.py --model_id $speaker_model_id --conf $conf_file \
#           --subseg_json $json_dir/subseg.json --embs_out $embs_dir --gpu $gpus --use_gpu
# fi
###### End extracting speaker embeddings ######

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "$(basename $0) Stage5: Perform clustering and postprocessing, and output sys rttms..."
  if [ "$include_overlap" = true ]; then
    cluster_rttm_dir=$rttm_dir/intermediate
  else
    cluster_rttm_dir=$rttm_dir
  fi
  torchrun --nproc_per_node=$nj local/cluster_and_postprocess.py --conf $conf_file --wavs $wav_list \
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
  torchrun --nproc_per_node=$nj local/out_transcription.py --exp_dir $exp --gpu $gpus
fi
