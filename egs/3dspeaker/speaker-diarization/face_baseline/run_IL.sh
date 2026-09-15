#!/bin/bash
#SBATCH --job-name=face-hmm-IL
#SBATCH --cpus-per-task=1
#SBATCH --output=face-hmm-IL-%j.log
set -eo pipefail

# Submit from the speaker-diarization directory, or set RECIPE_ROOT explicitly.
recipe_root="${RECIPE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if type module >/dev/null 2>&1; then
    module purge
    module load miniforge/24.1.2 cuda/11.8 gcc/9.3
    source activate 3D-Speaker
fi
cd "$recipe_root"
export PYTHONPATH="$recipe_root/../../..:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
tv_name="I love my family"
workspace_root="$(cd "$recipe_root/../../../.." && pwd)"
data_root="${DATA_ROOT:-$workspace_root/dataset}"
source_exp="${SOURCE_EXP:-$recipe_root/runs/$tv_name/exp_video_ablation}"
result_dir="${RESULT_DIR:-$source_exp/result/face_hmm}"
extra_args=()
if [[ -n "${AHC_LABELS:-}" ]]; then
    extra_args+=(--initial_labels "$AHC_LABELS")
fi
python local/cluster_and_postprocess_face.py \
    --conf "conf/$tv_name/diar_video.yaml" \
    --wavs "$data_root/$tv_name/raw/wav.list" \
    --visual_embs_dir "${VISUAL_EMBS_DIR:-$source_exp/embs_video}" \
    --subseg_json "${SUBSEG_JSON:-$source_exp/json/subseg_ori.json}" \
    --result_dir "$result_dir" "${extra_args[@]}"
python local/compute_acc_face.py --result_dir "$result_dir" \
    --ref_xlsx "$data_root/$tv_name/annotation/faces_annotation_with_loc_new.xlsx" \
    --mode "${EVAL_MODE:-all}"
