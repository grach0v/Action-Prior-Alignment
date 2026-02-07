#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU=0
MODEL_PATH="$ROOT/a2_pretrained/checkpoints/sl_checkpoint_199.pth"
LOG_SUFFIX="a2"
LOG_ROOT="$ROOT/logs"
PYTHON_BIN="$ROOT/.venv/bin/python"
PY_SITE="$ROOT/.venv/lib/python3.10/site-packages"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --log-suffix)
      LOG_SUFFIX="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

export PATH="$ROOT/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PY_SITE/torch/lib:${HOME}/cuda-13.0/lib64:$PY_SITE/nvidia/nccl/lib:$PY_SITE/nvidia/cusparselt/lib:$PY_SITE/nvidia/cusparse/lib:$PY_SITE/nvidia/cudnn/lib:$PY_SITE/nvidia/cuda_runtime/lib"

echo "$MODEL_PATH"
echo "seen"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m a2.evaluate.test_pickplace \
  --workspace_shift --use_rope --load_model --model_path "$MODEL_PATH" \
  --log_suffix "pp-${LOG_SUFFIX}" \
  --log_root "$LOG_ROOT" \
  --testing_case_dir "testing_cases/pp_testing_cases/seen"

echo "unseen"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m a2.evaluate.test_pickplace \
  --workspace_shift --unseen --use_rope --load_model --model_path "$MODEL_PATH" \
  --log_suffix "pp-${LOG_SUFFIX}-unseen" \
  --log_root "$LOG_ROOT" \
  --testing_case_dir "testing_cases/pp_testing_cases/unseen"
