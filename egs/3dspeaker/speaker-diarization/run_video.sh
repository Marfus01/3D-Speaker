#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script performs speaker diarization task based on video input.
# It extracts both visual and audio speaker embeddings and generates more accurate results than audio-only diarization.

set -e  # 如果脚本中的任何命令失败，脚本会立即退出。
. ./path.sh || exit 1 # 加载 path.sh 文件以设置必要的环境变量，并在加载失败时退出脚本。

stage=1 # 标识每个处理步骤的index
stop_stage=6  # 共6步
cluster_type="audio_vision" # 聚类方式，支持 "audio_only" 和 "audio_vision"

data_root=/f/data/tv_series_plus/tv_data # 存储所有电视剧数据集的根目录
tv_name="I love my family" # "the big bang theory", "I love my family"
language="zh-cn" # 语言类型，支持 "en" 和 "zh-cn"

from_subtitle=true  # 是否直接从字幕文件中提取说话人分割信息
onnx_dir=pretrained_models  # 存储预训练模型的目录
gpus="0"  # 指定可用的 GPU ID
nj=1  # 并行任务数
master_port=29567  # 用于分布式训练的主节点端口号
FFMPEG_PATH="/d/wangchen/useful_tools/ffmpeg/install/bin/ffmpeg.exe"

# HMM平滑相关参数
use_hmm_smoothing=true  # 在"audio_vision"聚类之后，是否做 HMM 平滑
fix_mf=false  # HMM平滑时，是否认为中间帧人脸聚类标签为ground truth
hmm_visual_info_type="vad+mid_frame"  # HMM平滑时，使用的视觉信息类型，支持 "", "vad", "mid_frame", "vad+mid_frame"
unreliable_pp=100.0  # HMM平滑时，认为不可靠的说话人标签百分比，范围0-100.0
hmm_flag=""
if [ "$use_hmm_smoothing" = true ]; then
  hmm_flag="--use_hmm_smoothing"
fi
fix_mf_flag=""
if [ "$fix_mf" = true ]; then
  fix_mf_flag="--fix_mf"
fi

# Self-supervised learning parameters
ft_flag=true  # 是否进行自监督微调
max_rounds=10  # 自监督学习的最大迭代轮数
finetune_lr=0.001  # 微调学习率
finetune_batch_size=64  # 微调batch size
warmup_epochs_num=2  # 分类器warmup轮数
finetune_epochs_num=5  # 每次微调的epoch数
unfrozen_layers_num=2  # 未冻结层的数量
early_stop_patience=5  # 早停patience


. local/parse_options.sh || exit 1  # 解析命令行参数，覆盖默认变量值

conf_file="conf/$tv_name/diar_video.yaml"
examples="$data_root/$tv_name" # 存储original video和说话人标注文件的目录
video_list=$examples/movie.list # 包含所有original video的路径
raw_data_dir=$examples/raw # 存储从original video中提取出的pure video和pure audio

exp="runs/$tv_name/exp_video" # 存储original video被处理后的所有中间文件和最终结果
visual_embs_dir=$exp/embs_video
result_dir=$exp/result  # 存储模型给出的说话人分离结果

if [[ "$cluster_type" != "audio_only" && "$cluster_type" != "audio_vision" ]]; then
  echo "Error: cluster_type must be either 'audio_only' or 'audio_vision'."
  exit 1
fi

# Set speaker_model_id to damo/speech_eres2net_sv_zh-cn_16k-common when using eres2net
if [ "$language" = "en" ]; then
  speaker_model_id=iic/speech_campplus_sv_en_voxceleb_16k
elif [ "$language" = "zh-cn" ]; then
  speaker_model_id=iic/speech_campplus_sv_zh-cn_3dspeaker_16k
else
  echo "Only support 'en' and 'zh-cn' for language now. Exit with error."
  exit 1
fi


if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then  # stage<=1 且 stop_stage>=1 时执行
  if [ ! -f "$video_list" ]; then
    echo "$(basename $0) Stage1: Prepare input videos..." # 下载完整视频，说话人标注和所有前两种文件的list
    # 获取 movie 目录下的所有 mkv 文件，按集数排序
    find "$examples/movie" -type f -name "E*.mkv" | sort > "$video_list"
    echo "Video list saved to $video_list"
  else
    echo "$(basename $0) Stage 1: $video_list exists. Skip this stage."
  fi
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  echo "$(basename $0) Stage2: Prepare onnx files and extrack raw videos and audios..."
  # Download pretrained models
  mkdir -p $onnx_dir
  for m in asd.onnx fqa.onnx face_recog_ir101.onnx; do
    if [ ! -e $onnx_dir/$m ]; then
      echo "$(basename $0) Stage2: Download pretrained models $m"
      wget -O $onnx_dir/$m "https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common/resolve/master/onnx/$m"
    fi
  done
  # Split each original video to pure video and pure audio(not segmented)
  mkdir -p "$raw_data_dir"
  cat "$video_list" | while read video_file; do
    filename=$(basename "$video_file")
    out_video_file=$raw_data_dir/${filename%.*}.mp4
    out_wav_file=$raw_data_dir/${filename%.*}.wav
    if [ ! -e "$out_video_file" ] && [ "$cluster_type" == "audio_vision" ]; then
      echo "$(basename "$0") Stage2: Extract video from $filename"
      $FFMPEG_PATH -nostdin -y -i "$video_file" -qscale:v 2 -threads 16 -async 1 -r 25 "$out_video_file" -loglevel panic
    fi
    if [ ! -e "$out_wav_file" ]; then
      echo "$(basename "$0") Stage2: Extract audio from $filename"
      $FFMPEG_PATH -nostdin -y -i "$out_video_file" -qscale:a 0 -ac 1 -vn -threads 16 -ar 16000 "$out_wav_file" -loglevel panic
    fi
  done
fi

# write two list, video.list and wav.list, which contain paths of all pure video/audios respectively
cat "$video_list" | while read video_file; do filename=$(basename "$video_file");echo "$raw_data_dir/${filename%.*}.mp4";done > "$raw_data_dir/video.list"
cat "$video_list" | while read video_file; do filename=$(basename "$video_file");echo "$raw_data_dir/${filename%.*}.wav";done > "$raw_data_dir/wav.list"

# use run_audio.sh to save audio speaker embeddings
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "$(basename "$0") Stage3: Extract audio speaker embeddings..."
  bash run_audio.sh --stage 2 --stop_stage 4 --from_subtitle $from_subtitle --speaker_model_id $speaker_model_id --examples "$raw_data_dir" --exp "$exp" --master_port $master_port --conf_file "$conf_file" --gpus $gpus --nj $nj
fi

# For each detected frame with one active speaker(with high quality face), record its timepoint and facial embedding in 'visual_embs_dir/{video_name}.pkl'
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && [ "$cluster_type" == "audio_vision" ]; then
  echo "$(basename $0) Stage4: Extract visual speaker embeddings..."
  mkdir -p "$exp/conf"
  cp "$conf_file" "$exp/conf/"  
  torchrun --nproc_per_node=$nj --master_port $master_port local/extract_visual_embeddings.py \
    --conf "$conf_file" --videos "$raw_data_dir/video.list" --vad "$exp/json/vad.json" --subseg "$exp/json/subseg.json"\
    --onnx_dir $onnx_dir --embs_out "$visual_embs_dir" --midframe_face_out "$examples/midframe_faces" --gpu $gpus --use_gpu
fi


if [ "$ft_flag" = false ]; then
  if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ "$cluster_type" == "audio_only" ]; then
      echo "$(basename $0) Stage5: Clustering for audio speaker embeddings only..."
      torchrun --nproc_per_node=$nj --master_port $master_port local/cluster_and_postprocess.py \
              --conf "$conf_file" --cluster_type "$cluster_type" --wavs "$raw_data_dir/wav.list" \
              --audio_embs_dir "$exp/embs" --result_dir "$result_dir" $hmm_flag
    else
      echo "$(basename $0) Stage5: Clustering for both type of speaker embeddings..."
      torchrun --nproc_per_node=$nj --master_port $master_port local/cluster_and_postprocess.py \
              --conf "$conf_file" --cluster_type "$cluster_type" --wavs "$raw_data_dir/wav.list" \
              --audio_embs_dir "$exp/embs" --visual_embs_dir "$visual_embs_dir" --result_dir "$result_dir" \
              $hmm_flag $fix_mf_flag --hmm_visual_info_type "$hmm_visual_info_type" --unreliable_pp $unreliable_pp
    fi
  fi

  if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "$(basename $0) Stage6: Get the final metrics..."
    speaker_anno_file=$examples/annotation/text_annotated.xlsx
    if [ -f "$speaker_anno_file" ]; then
      echo "Computing speaker recognition accuracy..."
      python local/compute_acc_spk.py --result_dir "$result_dir" --ref_xlsx "$speaker_anno_file"
    else
      echo "Speaker_anno_file "$speaker_anno_file" is not detected. Can't calculate the result"
    fi
    face_anno_file=$examples/annotation/faces_annotation_with_loc_new.xlsx
    if [ -f "$face_anno_file" ]; then
      echo "Computing face recognition accuracy..."
      python local/compute_acc_face.py --result_dir "$result_dir" --ref_xlsx "$face_anno_file"
    else
      echo "Face_anno_file "$face_anno_file" is not detected. Can't calculate the result"
    fi
  fi
else
  if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "$(basename $0) Stage5: Self-supervised fine-tuning..."
    speaker_anno_file=$examples/annotation/text_annotated.xlsx
    if [ ! -f "$speaker_anno_file" ]; then
      echo "Error: Speaker annotation file $speaker_anno_file not found. Self-supervised learning requires annotations for evaluation."
      exit 1
    fi
    
    # Run self-supervised fine-tuning
    torchrun --nproc_per_node=$nj --master_port $master_port local/self_supervised_finetune.py \
      --conf "$conf_file" --cluster_type "$cluster_type" --wavs "$raw_data_dir/wav.list" \
      --audio_embs_dir "$exp/embs" --visual_embs_dir "$visual_embs_dir" --result_dir "$result_dir" \
      $hmm_flag $fix_mf_flag --hmm_visual_info_type "$hmm_visual_info_type" --unreliable_pp $unreliable_pp \
      --speaker_anno_file "$speaker_anno_file" \
      --speaker_model_id "$speaker_model_id" \
      --subseg_json "$exp/json/subseg.json" \
      --max_rounds $max_rounds --finetune_lr $finetune_lr --finetune_batch_size $finetune_batch_size \
      --warmup_epochs_num $warmup_epochs_num --finetune_epochs_num $finetune_epochs_num \
      --unfrozen_layers_num $unfrozen_layers_num --early_stop_patience $early_stop_patience \
      --use_gpu --gpu $gpus \
      --seed 1234 \
    
    echo "$(basename $0) Stage5: Self-supervised fine-tuning completed!"
    echo "Results saved in $result_dir/self_supervised/"
    echo "Best model saved as $result_dir/self_supervised/best_model.pth"
    echo "Accuracy history saved in $result_dir/self_supervised/accuracy_history.json"
  fi
  if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "$(basename $0) Stage6: Visualize fine-tuning results..."
  fi
fi

# if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
#   echo "$(basename $0) Stage6: Get the final metrics..."
#   ref_rttm_list=$examples/refrttm.list
#   if [ -f $ref_rttm_list ]; then
#     cat $ref_rttm_list | while read line;do cat $line;done > $exp/concat_ref_rttm
#     echo "Computing DER..."
#     python local/compute_der.py --exp_dir $exp --ref_rttm $exp/concat_ref_rttm
#   else
#     echo "Refrttm.list is not detected. Can't calculate the result"
#   fi
# fi
