#!/bin/bash
# set -x
gpu=0

# make sure we use the local CUDA toolkit and uv venv
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export CUDA_HOME="${CUDA_HOME:-${REPO_ROOT}/.cuda-nvcc}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/.venv/bin/activate"

# model_path=a2_pretrained/checkpoints/sl_checkpoint_199.pth
model_path=logs/2025-12-10-13-25-05-train-a2/checkpoints/sl_checkpoint_2025-12-10_21-56-29_199.pth
log=a2_mytrained

echo $model_path
echo seen
CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_pick.py --use_rope --load_model --model_path $model_path --log_suffix grasp-$log --testing_case_dir testing_cases/grasp_testing_cases/seen
echo unseen
CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_pick.py --use_rope --load_model --model_path $model_path --log_suffix grasp-$log-unseen --testing_case_dir testing_cases/grasp_testing_cases/unseen
