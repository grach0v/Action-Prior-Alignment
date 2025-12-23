#!/bin/bash
# set -x
gpu=0

# Default checkpoint baked into the repo
model_path=logs/a2_pretrained/checkpoints/sl_checkpoint_199.pth
# model_path=logs/2025-12-10-13-25-05-train-a2/checkpoints/sl_checkpoint_2025-12-10_21-56-29_199.pth
log=a2_mytrained_new_grasp

echo $model_path
echo seen
CUDA_VISIBLE_DEVICES=$gpu uv run python -m a2.evaluate.test_pick --use_rope --load_model --model_path $model_path --log_suffix grasp-$log --testing_case_dir testing_cases/grasp_testing_cases/seen
echo unseen
CUDA_VISIBLE_DEVICES=$gpu uv run python -m a2.evaluate.test_pick --use_rope --load_model --model_path $model_path --log_suffix grasp-$log-unseen --testing_case_dir testing_cases/grasp_testing_cases/unseen
