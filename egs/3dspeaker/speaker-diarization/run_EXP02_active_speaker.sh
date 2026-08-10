#!/bin/bash

set -Ee -o pipefail

trap 'echo "[ERROR] EXP-02 job stopped at line ${LINENO}." >&2' ERR

recipe_root="/data02/home/scv7387/run/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-diarization"
analysis_script="$recipe_root/local/analyze_exp02_active_speaker.py"

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 \"the big bang theory\"|\"I love my family\"" >&2
    exit 2
fi

tv_name="$1"
case "$tv_name" in
    "the big bang theory"|"I love my family")
        ;;
    *)
        echo "[ERROR] Unsupported dataset: $tv_name" >&2
        exit 2
        ;;
esac

module purge
module load miniforge/24.1.2 cuda/11.8 gcc/9.3
source activate 3D-Speaker

data_root="/data02/home/scv7387/run/tv_series_plus/dataset"
device_id="${DEVICE_ID:-0}"
ffmpeg_bin="${FFMPEG_BIN:-ffmpeg}"
min_recovery_rate="${MIN_RECOVERY_RATE:-0.99}"

if [[ ! -f "$analysis_script" ]]; then
    echo "[ERROR] Analysis script not found: $analysis_script" >&2
    exit 1
fi

if ! command -v "$ffmpeg_bin" >/dev/null 2>&1 && [[ ! -x "$ffmpeg_bin" ]]; then
    echo "[ERROR] FFmpeg executable not found: $ffmpeg_bin" >&2
    exit 1
fi

cd "$recipe_root"

run_stage() {
    local stage_name="$1"
    shift
    echo "[INFO] Starting $stage_name"
    "$@"
    echo "[INFO] Finished $stage_name"
}

common_args=(
    python
    "$analysis_script"
    --data-root "$data_root"
    --tv-name "$tv_name"
)

run_stage "Stage 1/4: coverage and identity agreement" \
    "${common_args[@]}" \
    --task summarize \
    --require-reference-identity

run_stage "Stage 2/4: recover all active-face bboxes" \
    "${common_args[@]}" \
    --task recover-bbox \
    --device cuda \
    --device-id "$device_id"

run_stage "Stage 3/4: enforce bbox recovery gate and plot distributions" \
    "${common_args[@]}" \
    --task plot-bbox \
    --min-recovery-rate "$min_recovery_rate"

run_stage "Stage 4/4: write all media, review workbooks, and recommendations" \
    "${common_args[@]}" \
    --task candidates \
    --write-clips \
    --ffmpeg "$ffmpeg_bin"

echo "[INFO] EXP-02 completed successfully for: $tv_name"
