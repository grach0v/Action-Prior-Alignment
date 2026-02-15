#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/denis-office/lerobot_a2/A2_original}"
PYTHON_BIN="${PYTHON_BIN:-/home/denis-office/miniconda3/envs/a2_gpu/bin/python}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-$GPU_ID}"      # comma-separated list, e.g. "0,1"
MODEL_PATH="${MODEL_PATH:-a2_pretrained/checkpoints/sl_checkpoint_199.pth}"
ASSETS_DIR="${ASSETS_DIR:-${A2_ASSETS_ROOT:-assets}}"   # e.g. assets or assets_orig
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/repro_runs/$RUN_ID}"
START_FROM="${START_FROM:-pick_seen}"   # pick_seen|pick_unseen|place_seen|place_unseen|pp_seen|pp_unseen
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
# Also accept PARALLEL=1 for convenience.
PARALLEL_TASKS="${PARALLEL_TASKS:-${PARALLEL:-0}}"   # 1: run selected tasks concurrently; 0: run sequentially
ALLOW_SINGLE_GPU_PARALLEL="${ALLOW_SINGLE_GPU_PARALLEL:-0}"
DEBUG_VERBOSE="${DEBUG_VERBOSE:-0}"
DEBUG_LEVEL="${DEBUG_LEVEL:-0}"
DEBUG_LOG_DIR="${DEBUG_LOG_DIR:-$OUT_DIR/debug}"
TEXTURE_DEBUG="${TEXTURE_DEBUG:-0}"
A2_PLACE_VALID_ONLY="${A2_PLACE_VALID_ONLY:-1}"
A2_PP_PLACE_VALID_ONLY="${A2_PP_PLACE_VALID_ONLY:-1}"

mkdir -p "$OUT_DIR"
mkdir -p "$DEBUG_LOG_DIR"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: python not found at $PYTHON_BIN"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model checkpoint not found at $MODEL_PATH"
  exit 1
fi

for path in \
  "$ASSETS_DIR/simplified_objects" \
  "$ASSETS_DIR/unseen_objects" \
  assets/ur5e/ur5e.urdf \
  assets/workspace/workspace.urdf \
  testing_cases/grasp_testing_cases \
  testing_cases/place_testing_cases \
  testing_cases/pp_testing_cases; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path"
    exit 1
  fi
done

run_eval() {
  local name="$1"
  local gpu="$2"
  shift
  shift
  local status_file="$OUT_DIR/${name}.exit"
  local task_log="$OUT_DIR/${name}.log"
  : > "$status_file"
  local dbg_file=""
  if [[ "$DEBUG_VERBOSE" != "0" || "$DEBUG_LEVEL" != "0" ]]; then
    dbg_file="$DEBUG_LOG_DIR/${name}.a2dbg.log"
  fi
  echo "[$(date '+%F %T')] START $name (GPU=$gpu) log=$task_log"
  set +e
  A2_DEBUG_VERBOSE="$DEBUG_VERBOSE" \
  A2_DEBUG_LEVEL="$DEBUG_LEVEL" \
  A2_DEBUG_LOG_FILE="$dbg_file" \
  A2_TEXTURE_DEBUG="$TEXTURE_DEBUG" \
  A2_PLACE_VALID_ONLY="$A2_PLACE_VALID_ONLY" \
  A2_PP_PLACE_VALID_ONLY="$A2_PP_PLACE_VALID_ONLY" \
  A2_ASSETS_ROOT="$ASSETS_DIR" \
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$@" >"$task_log" 2>&1
  local status=$?
  set -e
  echo "$status" > "$status_file"
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAIL  $name (exit=$status) log=$task_log"
  else
    echo "[$(date '+%F %T')] DONE  $name log=$task_log"
  fi
  return "$status"
}

gpu_for_task() {
  local task_ord="$1"
  local gpu_csv="$2"
  IFS=',' read -r -a gpu_arr <<< "$gpu_csv"
  local n="${#gpu_arr[@]}"
  if [[ "$n" -eq 0 ]]; then
    echo "$GPU_ID"
    return
  fi
  local idx=$((task_ord % n))
  echo "${gpu_arr[$idx]}"
}

gpu_count() {
  local gpu_csv="$1"
  IFS=',' read -r -a gpu_arr <<< "$gpu_csv"
  local n=0
  local g
  for g in "${gpu_arr[@]}"; do
    if [[ -n "${g// }" ]]; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

if [[ "$PARALLEL_TASKS" == "1" ]]; then
  n_gpu="$(gpu_count "$GPU_IDS")"
  if [[ "$n_gpu" -lt 2 && "$ALLOW_SINGLE_GPU_PARALLEL" != "1" ]]; then
    echo "WARN: PARALLEL_TASKS=1 with GPU_IDS=$GPU_IDS (single GPU) can hurt reproducibility/perf."
    echo "WARN: Auto-switching to sequential mode. Set ALLOW_SINGLE_GPU_PARALLEL=1 to force."
    PARALLEL_TASKS=0
  fi
fi

find_session_dir() {
  local suffix="$1"
  ls -dt logs/*-"${suffix}" 2>/dev/null | head -n1 || true
}

summarize_session() {
  local tag="$1"
  local suffix="$2"
  local session_dir
  local summary_file="$OUT_DIR/${tag}_summary.txt"
  session_dir="$(find_session_dir "$suffix")"
  if [[ -z "${session_dir:-}" ]]; then
    echo "WARN: no session directory found for suffix $suffix"
    return 0
  fi
  "$PYTHON_BIN" helpers/evaluate_log.py --session_directory "$session_dir" | tee "$summary_file" >/dev/null
  echo "$session_dir" > "$OUT_DIR/${tag}_session_dir.txt"
}

task_index() {
  case "$1" in
    pick_seen) echo 1 ;;
    pick_unseen) echo 2 ;;
    place_seen) echo 3 ;;
    place_unseen) echo 4 ;;
    pp_seen) echo 5 ;;
    pp_unseen) echo 6 ;;
    *)
      echo "ERROR: invalid task name '$1'" >&2
      exit 1
      ;;
  esac
}

should_run() {
  local task="$1"
  [[ "$(task_index "$task")" -ge "$(task_index "$START_FROM")" ]]
}

pick_seen_suffix="grasp-a2-gpu-seen-${RUN_ID}"
pick_unseen_suffix="grasp-a2-gpu-unseen-${RUN_ID}"
place_seen_suffix="place-a2-gpu-seen-${RUN_ID}"
place_unseen_suffix="place-a2-gpu-unseen-${RUN_ID}"
pp_seen_suffix="pp-a2-gpu-seen-${RUN_ID}"
pp_unseen_suffix="pp-a2-gpu-unseen-${RUN_ID}"
FAILED_TASKS=()
RUN_PIDS=()
RUN_NAMES=()

launch_eval() {
  local name="$1"
  shift
  run_eval "$name" "$@" &
  RUN_PIDS+=("$!")
  RUN_NAMES+=("$name")
}

collect_parallel_statuses() {
  local i
  for i in "${!RUN_PIDS[@]}"; do
    local pid="${RUN_PIDS[$i]}"
    local name="${RUN_NAMES[$i]}"
    set +e
    wait "$pid"
    set -e
    local status_file="$OUT_DIR/${name}.exit"
    local status="1"
    if [[ -f "$status_file" ]]; then
      status="$(cat "$status_file")"
    fi
    if [[ "$status" -ne 0 ]]; then
      FAILED_TASKS+=("$name")
    fi
  done
}

if should_run "pick_seen"; then
  pick_seen_gpu="$(gpu_for_task 0 "$GPU_IDS")"
  if [[ "$PARALLEL_TASKS" == "1" ]]; then
    launch_eval "pick_seen" "$pick_seen_gpu" \
      a2/evaluate/test_pick.py \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$pick_seen_suffix" \
      --testing_case_dir testing_cases/grasp_testing_cases/seen
  else
    if ! run_eval "pick_seen" "$pick_seen_gpu" \
      a2/evaluate/test_pick.py \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$pick_seen_suffix" \
      --testing_case_dir testing_cases/grasp_testing_cases/seen; then
      FAILED_TASKS+=("pick_seen")
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit 1; fi
    fi
  fi
fi

if should_run "pick_unseen"; then
  pick_unseen_gpu="$(gpu_for_task 1 "$GPU_IDS")"
  if [[ "$PARALLEL_TASKS" == "1" ]]; then
    launch_eval "pick_unseen" "$pick_unseen_gpu" \
      a2/evaluate/test_pick.py \
      --unseen \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$pick_unseen_suffix" \
      --testing_case_dir testing_cases/grasp_testing_cases/unseen
  else
    if ! run_eval "pick_unseen" "$pick_unseen_gpu" \
      a2/evaluate/test_pick.py \
      --unseen \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$pick_unseen_suffix" \
      --testing_case_dir testing_cases/grasp_testing_cases/unseen; then
      FAILED_TASKS+=("pick_unseen")
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit 1; fi
    fi
  fi
fi

if should_run "place_seen"; then
  place_seen_gpu="$(gpu_for_task 2 "$GPU_IDS")"
  if [[ "$PARALLEL_TASKS" == "1" ]]; then
    launch_eval "place_seen" "$place_seen_gpu" \
      a2/evaluate/test_place.py \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$place_seen_suffix" \
      --testing_case_dir testing_cases/place_testing_cases/seen \
      --action_var
  else
    if ! run_eval "place_seen" "$place_seen_gpu" \
      a2/evaluate/test_place.py \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$place_seen_suffix" \
      --testing_case_dir testing_cases/place_testing_cases/seen \
      --action_var; then
      FAILED_TASKS+=("place_seen")
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit 1; fi
    fi
  fi
fi

if should_run "place_unseen"; then
  place_unseen_gpu="$(gpu_for_task 3 "$GPU_IDS")"
  if [[ "$PARALLEL_TASKS" == "1" ]]; then
    launch_eval "place_unseen" "$place_unseen_gpu" \
      a2/evaluate/test_place.py \
      --unseen \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$place_unseen_suffix" \
      --testing_case_dir testing_cases/place_testing_cases/unseen \
      --action_var
  else
    if ! run_eval "place_unseen" "$place_unseen_gpu" \
      a2/evaluate/test_place.py \
      --unseen \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$place_unseen_suffix" \
      --testing_case_dir testing_cases/place_testing_cases/unseen \
      --action_var; then
      FAILED_TASKS+=("place_unseen")
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit 1; fi
    fi
  fi
fi

if should_run "pp_seen"; then
  pp_seen_gpu="$(gpu_for_task 4 "$GPU_IDS")"
  if [[ "$PARALLEL_TASKS" == "1" ]]; then
    launch_eval "pp_seen" "$pp_seen_gpu" \
      a2/evaluate/test_pickplace.py \
      --workspace_shift \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$pp_seen_suffix" \
      --testing_case_dir testing_cases/pp_testing_cases/seen
  else
    if ! run_eval "pp_seen" "$pp_seen_gpu" \
      a2/evaluate/test_pickplace.py \
      --workspace_shift \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$pp_seen_suffix" \
      --testing_case_dir testing_cases/pp_testing_cases/seen; then
      FAILED_TASKS+=("pp_seen")
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit 1; fi
    fi
  fi
fi

if should_run "pp_unseen"; then
  pp_unseen_gpu="$(gpu_for_task 5 "$GPU_IDS")"
  if [[ "$PARALLEL_TASKS" == "1" ]]; then
    launch_eval "pp_unseen" "$pp_unseen_gpu" \
      a2/evaluate/test_pickplace.py \
      --workspace_shift \
      --unseen \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$pp_unseen_suffix" \
      --testing_case_dir testing_cases/pp_testing_cases/unseen
  else
    if ! run_eval "pp_unseen" "$pp_unseen_gpu" \
      a2/evaluate/test_pickplace.py \
      --workspace_shift \
      --unseen \
      --use_rope \
      --load_model \
      --model_path "$MODEL_PATH" \
      --log_suffix "$pp_unseen_suffix" \
      --testing_case_dir testing_cases/pp_testing_cases/unseen; then
      FAILED_TASKS+=("pp_unseen")
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit 1; fi
    fi
  fi
fi

if [[ "$PARALLEL_TASKS" == "1" ]]; then
  collect_parallel_statuses
  if [[ "$CONTINUE_ON_ERROR" != "1" && "${#FAILED_TASKS[@]}" -gt 0 ]]; then
    echo "ERROR: one or more parallel tasks failed and CONTINUE_ON_ERROR=0"
    exit 1
  fi
fi

summarize_session "pick_seen" "$pick_seen_suffix"
summarize_session "pick_unseen" "$pick_unseen_suffix"
summarize_session "place_seen" "$place_seen_suffix"
summarize_session "place_unseen" "$place_unseen_suffix"
summarize_session "pp_seen" "$pp_seen_suffix"
summarize_session "pp_unseen" "$pp_unseen_suffix"

print_task_logs() {
  echo
  echo "Task logs:"
  for tag in pick_seen pick_unseen place_seen place_unseen pp_seen pp_unseen; do
    local log_file="$OUT_DIR/${tag}.log"
    if [[ -f "$log_file" ]]; then
      echo "- $tag: $log_file"
    fi
  done
}

extract_metric() {
  local file="$1"
  local key="$2"
  awk -F': ' -v k="$key" '$1==k {print $2}' "$file" | tail -n1
}

report_file="$OUT_DIR/report.md"
{
  echo "# A2 Pretrained Reproduction Report"
  echo
  echo "- Run ID: \`$RUN_ID\`"
  echo "- Repo: \`$REPO_ROOT\`"
  echo "- Python: \`$PYTHON_BIN\`"
  echo "- GPU ID: \`$GPU_ID\`"
  echo "- GPU IDs: \`$GPU_IDS\`"
  echo "- Parallel Tasks: \`$PARALLEL_TASKS\`"
  echo "- Model: \`$MODEL_PATH\`"
  echo "- Assets Dir: \`$ASSETS_DIR\`"
  echo "- A2_SAMPLE_NUM_GRASP: \`${A2_SAMPLE_NUM_GRASP:-}\`"
  echo "- A2_SAMPLE_NUM_PLACE: \`${A2_SAMPLE_NUM_PLACE:-}\`"
  echo "- A2_SAMPLE_NUM_PLACE_IN: \`${A2_SAMPLE_NUM_PLACE_IN:-}\`"
  echo "- A2_SAMPLE_NUM_PLACE_OUT: \`${A2_SAMPLE_NUM_PLACE_OUT:-}\`"
  echo "- A2_PLACE_RELATION_ADAPTIVE: \`${A2_PLACE_RELATION_ADAPTIVE:-}\`"
  echo "- A2_PLACE_SAMPLE_PER_OBJ: \`${A2_PLACE_SAMPLE_PER_OBJ:-}\`"
  echo "- A2_PLACE_RELATION_PRIOR_WEIGHT: \`${A2_PLACE_RELATION_PRIOR_WEIGHT:-}\`"
  echo "- A2_MOVE_JOINTS_TIMEOUT: \`${A2_MOVE_JOINTS_TIMEOUT:-}\`"
  echo "- A2_PLACE_VALID_ONLY: \`${A2_PLACE_VALID_ONLY:-}\`"
  echo "- A2_PP_PLACE_VALID_ONLY: \`${A2_PP_PLACE_VALID_ONLY:-}\`"
  echo "- A2_PP_TARGET_WORKSPACE_MARGIN: \`${A2_PP_TARGET_WORKSPACE_MARGIN:-}\`"
  echo "- A2_PP_DROP_OUT_OF_WORKSPACE_CLUTTER: \`${A2_PP_DROP_OUT_OF_WORKSPACE_CLUTTER:-}\`"
  echo
  echo "## Aggregated Metrics"
  echo
  echo "| Task | Split | Avg Task Success (%) | Avg Step | Avg Success Step | Session Dir |"
  echo "|---|---:|---:|---:|---:|---|"
  for tag in pick_seen pick_unseen place_seen place_unseen pp_seen pp_unseen; do
    summary="$OUT_DIR/${tag}_summary.txt"
    task="${tag%_*}"
    split="${tag##*_}"
    if [[ -f "$summary" && -f "$OUT_DIR/${tag}_session_dir.txt" ]]; then
      session_dir="$(cat "$OUT_DIR/${tag}_session_dir.txt")"
      success="$(extract_metric "$summary" "Average Task Success")"
      avg_step="$(extract_metric "$summary" "Average Step")"
      avg_success_step="$(extract_metric "$summary" "Average Success Step")"
      echo "| ${task} | ${split} | ${success} | ${avg_step} | ${avg_success_step} | \`${session_dir}\` |"
    else
      echo "| ${task} | ${split} | N/A | N/A | N/A | skipped or failed |"
    fi
  done
  echo
  echo "## Paper Targets (Table I, A2)"
  echo
  echo "- Pick Seen: 95.3 / 2.55"
  echo "- Pick Unseen: 97.3 / 2.57"
  echo "- Place Seen: 89.3"
  echo "- Place Unseen: 74.0"
  echo "- Pick-n-Place Seen: 87.5 / 2.45"
  echo "- Pick-n-Place Unseen: 71.7 / 3.02"
} > "$report_file"

if [[ "${#FAILED_TASKS[@]}" -gt 0 ]]; then
  {
    echo
    echo "## Failed Tasks"
    for task in "${FAILED_TASKS[@]}"; do
      echo "- $task"
    done
  } >> "$report_file"
fi

echo
echo "Done. Outputs:"
echo "- Run directory: $OUT_DIR"
echo "- Final report: $report_file"
print_task_logs
