#!/bin/bash
# set -x
gpu=0
data_path="data/a2_pp_data.npy"
log_suffix=a2

train () {
  CUDA_VISIBLE_DEVICES=$gpu python -m a2.train.main \
    --data_path "$data_path" \
    --log_suffix "$log_suffix" \
    "$@"
}

train --adjust_lr --efficient_attn mobile --heads 8 --hidden_size 256 --lang_emb --layers 3 --lr 0.000038 --step_ratio 0.483219 --step_size 45 --width 768
train --efficient_attn mobile --heads 4 --hidden_size 512 --lang_emb --layers 3 --lr 0.000071	--normalize --width 768
train --adjust_lr --heads 4 --hidden_size 384 --lang_emb --layers 1 --lr 0.000137	--step_ratio 0.144732 --step_size 25 --width 640
train --heads 4 --hidden_size 256 --lang_emb --layers 2 --lr 0.000098 --normalize --width 512
train --adjust_lr --efficient_attn efficient --heads 4 --hidden_size 256 --lang_emb --layers 1 --lr 0.000127 --step_ratio 0.800682	--step_size 35 --width 640