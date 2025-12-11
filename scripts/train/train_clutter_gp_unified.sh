#!/bin/bash
# set -x
gpu=1

# make sure we use the local CUDA toolkit and uv venv
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export CUDA_HOME="${CUDA_HOME:-${REPO_ROOT}/.cuda-nvcc}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/.venv/bin/activate"

data_path=data/a2_pp_data.npy
log_suffix=a2

CUDA_VISIBLE_DEVICES=$gpu python a2/train/main.py --lr 1e-4 --use_rope --data_path $data_path --log_suffix $log_suffix
