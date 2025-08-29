#!/bin/bash
# set -x
gpu=0

# model_path=logs/a2_pretrained/checkpoints/sl_checkpoint_199.pth
# model_path=logs/2025-07-20-17-52-02-train-a2/checkpoints/sl_checkpoint_2025-07-20_23-01-50_199.pth
# model_path=logs/2025-07-28-12-08-39-train-a2/checkpoints/sl_checkpoint_2025-07-28_18-39-32_249.pth
# model_path=logs/2025-08-12-17-22-10-train-a2/checkpoints/sl_checkpoint_2025-08-12_20-54-07_249.pth
# model_path=logs/2025-08-12-17-22-10-train-a2/checkpoints/sl_checkpoint_2025-08-12_20-12-16_199.pth
# model_path=logs/2025-08-26-11-59-07-train-a2/checkpoints/sl_checkpoint_2025-08-26_17-55-30_249.pth
# model_path=logs/2025-08-26-11-59-07-train-a2/checkpoints/sl_checkpoint_2025-08-26_14-22-45_99.pth
model_path=logs/2025-08-26-11-59-07-train-a2/checkpoints/sl_checkpoint_2025-08-26_13-11-39_49.pth
log=a2

echo $model_path
echo seen 
CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_pick.py --use_rope --load_model --model_path $model_path --log_suffix grasp-$log --testing_case_dir testing_cases/grasp_testing_cases/seen --efficient_attn --normalize --layers 2
echo unseen
CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_pick.py --use_rope --load_model --model_path $model_path --log_suffix grasp-$log-unseen --testing_case_dir testing_cases/grasp_testing_cases/unseen --efficient_attn --normalize --layers 2
