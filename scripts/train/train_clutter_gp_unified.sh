#!/bin/bash
# set -x
gpu=0

data_path=data/a2_pp_data.npy
log_suffix=a2

CUDA_VISIBLE_DEVICES=$gpu python a2/train/main.py --lr 1e-4 --use_rope --data_path $data_path --log_suffix $log_suffix --epoch_num 250
