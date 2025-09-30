#!/bin/bash

# 设置 Conda 环境变量
CONDA_ENV_NAME="tv_hmm"
CONDA_PATH="/d/ProgramFiles/anaconda3"

# 确保 Conda 的路径正确
if [ ! -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
  echo "Conda not found at $CONDA_PATH"
  exit 1
fi

# 加载 Conda 环境
source "$CONDA_PATH/etc/profile.d/conda.sh" || { echo "Failed to source conda.sh"; exit 1; }

# 激活 Conda 环境
conda activate "$CONDA_ENV_NAME" || { echo "Failed to activate Conda environment: $CONDA_ENV_NAME"; exit 1; }

echo "Conda environment '$CONDA_ENV_NAME' activated successfully."