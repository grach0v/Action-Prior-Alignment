#!/bin/bash
set -e

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU=0
DATA_PATH="$ROOT/data/a2_place_adaptive_data.npy"
MODEL_PATH="$ROOT/a2_pretrained/checkpoints/sl_checkpoint_199.pth"
LOG_SUFFIX="a2-adaptive-repro"
LOG_ROOT="$ROOT/logs/train/runs"
LR=1e-5
SAMPLE_NUM=100
PYTHON_BIN="$ROOT/.venv/bin/python"
PY_SITE="$ROOT/.venv/lib/python3.10/site-packages"

if [ ! -f "$DATA_PATH" ]; then
  echo "Missing adaptive dataset: $DATA_PATH"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing model checkpoint: $MODEL_PATH"
  exit 1
fi

export PATH="$ROOT/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PY_SITE/torch/lib:${HOME}/cuda-13.0/lib64:$PY_SITE/nvidia/nccl/lib:$PY_SITE/nvidia/cusparselt/lib:$PY_SITE/nvidia/cusparse/lib:$PY_SITE/nvidia/cudnn/lib:$PY_SITE/nvidia/cuda_runtime/lib"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -m a2.train.main \
  --lr "$LR" \
  --sample_num "$SAMPLE_NUM" \
  --use_rope \
  --adaptive \
  --data_path "$DATA_PATH" \
  --load_model \
  --model_path "$MODEL_PATH" \
  --log_suffix "$LOG_SUFFIX" \
  --log_root "$LOG_ROOT" \
  "$@"
