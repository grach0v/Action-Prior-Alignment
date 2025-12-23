#!/bin/bash
# set -x
gpu=0

data_path=data/a2_pp_data.npy
log_suffix=a2_new_grasp_my_train

CUDA_VISIBLE_DEVICES=$gpu uv run python -m a2.train.main --lr 1e-4 --use_rope --data_path $data_path --log_suffix $log_suffix --epoch_num 250
