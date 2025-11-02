#!/bin/bash
# set -x
gpu=${CUDA_DEVICE_ID:-0}

model_path=a2_pretrained/checkpoints/sl_checkpoint_199.pth
log=a2

echo $model_path
echo seen
CUDA_VISIBLE_DEVICES=$gpu python -m a2.evaluate.test_pickplace --workspace_shift --use_rope --load_model --model_path $model_path --log_suffix pp-$log --testing_case_dir testing_cases/pp_testing_cases/seen
echo unseen
CUDA_VISIBLE_DEVICES=$gpu python -m a2.evaluate.test_pickplace --workspace_shift --unseen --use_rope --load_model --model_path $model_path --log_suffix pp-$log-unseen --testing_case_dir testing_cases/pp_testing_cases/unseen
