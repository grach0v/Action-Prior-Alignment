#!/bin/bash
# set -x
gpu=7

data_path=data/a2_pp_data.npy
log_suffix=a2_default

CUDA_VISIBLE_DEVICES=$gpu python a2/train/main.py --lr 1e-4 --use_rope --data_path $data_path --log_suffix $log_suffix
