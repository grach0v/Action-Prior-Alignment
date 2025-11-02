#!/bin/bash
# set -x
gpu=0

echo collect data for grasp
CUDA_VISIBLE_DEVICES=$gpu python -m data_collection.collect_data_grasp --log_suffix collect-data-grasp
