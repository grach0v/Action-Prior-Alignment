#!/bin/bash
# set -x
gpu=7

data_path=data/a2_pp_data.npy
# log_suffix=a2_default

# CUDA_VISIBLE_DEVICES=$gpu python a2/train/main.py --lr 1e-4 --use_rope --data_path "$data_path" --log_suffix "$log_suffix"

efficient_attn_list=$(python - <<'PY'
from models.efficient_attention import list_efficient_attention_choices
print(" ".join(list_efficient_attention_choices(include_none=False)))
PY
)

for attn in $efficient_attn_list; do
  CUDA_VISIBLE_DEVICES=$gpu python a2/train/main.py --lr 1e-4 --use_rope --data_path "$data_path" --efficient_attn "$attn" --"log_suffix" "$efficient_attention_${attn}"
done
