#!/bin/bash
# set -x
gpu=0

# model_path=logs/a2_pretrained/checkpoints/sl_checkpoint_199.pth
# model_path=logs/2025-07-20-17-52-02-train-a2/checkpoints/sl_checkpoint_2025-07-20_23-01-50_199.pth
model_path=logs/2025-07-28-12-08-39-train-a2/checkpoints/sl_checkpoint_2025-07-28_18-39-32_249.pth
log=a2

echo $model_path
echo seen
CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_pick.py --use_rope --load_model --model_path $model_path --log_suffix grasp-$log --testing_case_dir testing_cases/grasp_testing_cases/seen
echo unseen
CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_pick.py --use_rope --load_model --model_path $model_path --log_suffix grasp-$log-unseen --testing_case_dir testing_cases/grasp_testing_cases/unseen
