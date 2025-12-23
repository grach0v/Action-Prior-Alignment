#!/bin/bash
# set -x
gpu=0

output_path=data/a2_pp_data.npy
record_dir=data/a2_pp_frames

echo "collect grasp+place data (recreate a2_pp_data.npy)"
CUDA_VISIBLE_DEVICES=$gpu uv run python data_collection/collect_data_pp.py --output_path "$output_path" --record_dir "$record_dir" "$@"
