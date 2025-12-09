#!/bin/bash
# set -x
gpu=0
source venv_py_39/bin/activate
echo "Using python: $(which python)"


checkpoints=(
"logs/2025-11-28-22-45-51-train-efficient_attention_sima/checkpoints/sl_checkpoint_2025-11-29_02-21-06_199.pth"
"logs/2025-11-28-17-56-30-train-efficient_attention_pola/checkpoints/sl_checkpoint_2025-11-28_22-45-43_199.pth"
"logs/2025-11-28-12-57-34-train-efficient_attention_mobile/checkpoints/sl_checkpoint_2025-11-28_17-56-19_199.pth"
"logs/2025-11-28-09-01-04-train-efficient_attention_linear/checkpoints/sl_checkpoint_2025-11-28_12-57-21_199.pth"
"logs/2025-11-28-06-00-23-train-efficient_attention_hydra/checkpoints/sl_checkpoint_2025-11-28_06-54-37_59.pth"
"logs/2025-11-28-06-00-23-train-efficient_attention_hydra/checkpoints/sl_checkpoint_2025-11-28_09-00-56_199.pth"
"logs/2025-11-28-01-54-03-train-efficient_attention_fast/checkpoints/sl_checkpoint_2025-11-28_06-00-15_199.pth"
"logs/2025-11-27-21-59-17-train-efficient_attention_efficient/checkpoints/sl_checkpoint_2025-11-28_01-53-55_199.pth"
"logs/2025-11-27-18-11-26-train-a2_default/checkpoints/sl_checkpoint_2025-11-27_21-59-08_199.pth"
)

for model_path in "${checkpoints[@]}"; do
    echo "================================================================================"
    echo "Processing $model_path"
    
    # Get config path (parent of parent of checkpoint)
    config_path=$(dirname $(dirname "$model_path"))/config.txt
    
    if [ ! -f "$config_path" ]; then
        echo "Config file not found at $config_path"
        continue
    fi

    # Parse config using python
    args=$(python3 -c "
import sys
import re

class device:
    def __init__(self, type):
        pass

try:
    with open('$config_path', 'r') as f:
        content = f.read()
        # Handle device object in string
        config = eval(content)
        
    cmd_args = []
    
    # String args
    if 'log_suffix' in config: cmd_args.append(f'--log_suffix grasp-{config[\"log_suffix\"]}')
    if 'efficient_attn' in config and config['efficient_attn']: cmd_args.append(f'--efficient_attn {config[\"efficient_attn\"]}')
    if 'lang_enc' in config: cmd_args.append(f'--lang_enc {config[\"lang_enc\"]}')
    if 'adaptive_type' in config: cmd_args.append(f'--adaptive_type {config[\"adaptive_type\"]}')
    if 'feat_backbone' in config: cmd_args.append(f'--feat_backbone {config[\"feat_backbone\"]}')
    
    # Int/Float args
    if 'width' in config: cmd_args.append(f'--width {config[\"width\"]}')
    if 'layers' in config: cmd_args.append(f'--layers {config[\"layers\"]}')
    if 'heads' in config: cmd_args.append(f'--heads {config[\"heads\"]}')
    if 'hidden_size' in config: cmd_args.append(f'--hidden_size {config[\"hidden_size\"]}')
    
    # Boolean args (flags)
    if config.get('fusion_sa'): cmd_args.append('--fusion_sa')
    if config.get('layer_norm'): cmd_args.append('--layer_norm')
    if config.get('lang_emb'): cmd_args.append('--lang_emb')
    if config.get('task_emb'): cmd_args.append('--task_emb')
    if config.get('use_rope'): cmd_args.append('--use_rope')
    if config.get('no_feat_rope'): cmd_args.append('--no_feat_rope')
    if config.get('no_rgb_feat'): cmd_args.append('--no_rgb_feat')
    if config.get('adaptive'): cmd_args.append('--adaptive')
    if config.get('normalize'): cmd_args.append('--normalize')
    
    print(' '.join(cmd_args))
except Exception as e:
    print(f'Error parsing config: {e}', file=sys.stderr)
    sys.exit(1)
")
    
    if [ $? -ne 0 ]; then
        echo "Failed to parse config for $model_path"
        continue
    fi
    
    echo "Extracted args: $args"

    # Run Seen
    echo "Running Seen Test..."
    CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_pick.py --load_model --model_path "$model_path" --testing_case_dir testing_cases/grasp_testing_cases/seen $args
    
    # Copy checkpoint to the latest log dir
    latest_log=$(ls -td logs/*-test-grasp-*/ | head -1)
    if [ -d "$latest_log" ]; then
        echo "Copying checkpoint to $latest_log/checkpoints/"
        mkdir -p "$latest_log/checkpoints/"
        cp "$model_path" "$latest_log/checkpoints/"
    else
        echo "Warning: Could not find latest log directory to copy checkpoint."
    fi

    # Run Unseen
    # Replace log_suffix for unseen
    args_unseen=$(echo "$args" | sed -E 's/--log_suffix ([^ ]+)/--log_suffix \1-unseen/')
    
    echo "Running Unseen Test..."
    CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_pick.py --load_model --model_path "$model_path" --testing_case_dir testing_cases/grasp_testing_cases/unseen $args_unseen
    
    latest_log=$(ls -td logs/*-test-grasp-*/ | head -1)
    if [ -d "$latest_log" ]; then
        echo "Copying checkpoint to $latest_log/checkpoints/"
        mkdir -p "$latest_log/checkpoints/"
        cp "$model_path" "$latest_log/checkpoints/"
    else
        echo "Warning: Could not find latest log directory to copy checkpoint."
    fi
    
done
