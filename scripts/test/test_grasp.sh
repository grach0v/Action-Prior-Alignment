#!/bin/bash
# set -x
gpu=0
log=a2
cases_seen="testing_cases/grasp_testing_cases/seen"
cases_unseen="testing_cases/grasp_testing_cases/unseen"
max_models=5

get_model_args() {
  local ckpt_path="$1"
  python - "$ckpt_path" <<'PY'
import ast
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
config_path = ckpt.parent.parent / "config.txt"
if not config_path.exists():
    sys.exit(0)

text = config_path.read_text()
safe_globals = {
    "__builtins__": None,
    "device": lambda **kwargs: kwargs.get("type"),
    "True": True,
    "False": False,
    "None": None,
}
try:
    cfg = eval(text, safe_globals, {})
except Exception:
    sys.exit(0)

flag_map = {
    "lang_emb": "--lang_emb",
    "normalize": "--normalize",
    "use_rope": "--use_rope",
    "fusion_sa": "--fusion_sa",
    "layer_norm": "--layer_norm",
    "task_emb": "--task_emb",
    "adaptive": "--adaptive",
    "no_feat_rope": "--no_feat_rope",
    "no_rgb_feat": "--no_rgb_feat",
}

value_map = {
    "heads": "--heads",
    "layers": "--layers",
    "hidden_size": "--hidden_size",
    "width": "--width",
    "ratio": "--ratio",
    "feat_backbone": "--feat_backbone",
    "lang_enc": "--lang_enc",
}

args = []
for key, flag in flag_map.items():
    if bool(cfg.get(key)):
        args.append(flag)

for key, flag in value_map.items():
    value = cfg.get(key)
    if value is not None:
        args.extend([flag, str(value)])

print(" ".join(args))
PY
}

mapfile -t all_models < <(ls -1t logs/*train-a2/checkpoints/sl_checkpoint_*.pth 2>/dev/null)

if [ ${#all_models[@]} -eq 0 ]; then
  echo "No checkpoints found under logs/*train-a2/checkpoints/"
  exit 1
fi

models=("${all_models[@]:0:max_models}")
echo "Testing ${#models[@]} model(s):"
printf ' - %s\n' "${models[@]}"

index=1
for model_path in "${models[@]}"; do
  run_name=$(basename "$(dirname "$(dirname "$model_path")")")
  log_suffix_seen="grasp-${log}-seen-${index}"
  log_suffix_unseen="grasp-${log}-unseen-${index}"

  eval_args_str=$(get_model_args "$model_path")
  if [ -n "$eval_args_str" ]; then
    read -r -a eval_args <<< "$eval_args_str"
  else
    eval_args=()
  fi

  echo "[$index] $run_name (seen)"
  CUDA_VISIBLE_DEVICES=$gpu python -m a2.evaluate.test_pick \
    --load_model \
    --model_path "$model_path" \
    --log_suffix "$log_suffix_seen" \
    --testing_case_dir "$cases_seen" \
    "${eval_args[@]}"

  echo "[$index] $run_name (unseen)"
  CUDA_VISIBLE_DEVICES=$gpu python -m a2.evaluate.test_pick \
    --load_model \
    --model_path "$model_path" \
    --log_suffix "$log_suffix_unseen" \
    --testing_case_dir "$cases_unseen" \
    "${eval_args[@]}"

  index=$((index + 1))
done
