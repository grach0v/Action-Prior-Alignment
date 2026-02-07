#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

NAME=""
GPU="0"
MODE="all" # train | benchmark | all
PARALLEL="1"
EPOCH_NUM="200"
MODEL_PATH=""
PYTHON_BIN="$ROOT/.venv/bin/python"

DATA_PATH="$ROOT/data/a2_pp_data.npy"
TRAIN_LR="1e-4"
TRAIN_SAMPLE_NUM="500"

usage() {
  cat <<EOF
Usage:
  bash scripts/experiments/train_and_benchmark_unified.sh --name <experiment_name> [options]

Options:
  --name <name>            Experiment name (required)
  --gpu <id>               GPU id for train+benchmark (default: 0)
  --mode <train|benchmark|all>
  --parallel <0|1>         Benchmark parallel mode (default: 1)
  --epochs <n>             Training epochs (default: 200)
  --model-path <path>      Use existing model for benchmark mode
  --python-bin <path>      Python executable (default: .venv/bin/python)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --epochs)
      EPOCH_NUM="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$NAME" ]]; then
  echo "Missing required argument: --name"
  usage
  exit 1
fi

if [[ "$MODE" != "train" && "$MODE" != "benchmark" && "$MODE" != "all" ]]; then
  echo "Invalid mode: $MODE"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

EXP_ROOT="$ROOT/logs/experiments/$NAME"
TRAIN_RUNS_ROOT="$EXP_ROOT/train/runs"
BENCH_RUNS_ROOT="$EXP_ROOT/benchmark/runs"
BENCH_CONSOLE_DIR="$EXP_ROOT/benchmark/console"

mkdir -p "$TRAIN_RUNS_ROOT" "$BENCH_RUNS_ROOT" "$BENCH_CONSOLE_DIR"

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
  if [[ ! -f "$DATA_PATH" ]]; then
    echo "Missing dataset: $DATA_PATH"
    exit 1
  fi

  TRAIN_SUFFIX="${NAME}-unified"
  echo "[train] name=$NAME suffix=$TRAIN_SUFFIX gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m a2.train.main \
    --lr "$TRAIN_LR" \
    --use_rope \
    --data_path "$DATA_PATH" \
    --sample_num "$TRAIN_SAMPLE_NUM" \
    --epoch_num "$EPOCH_NUM" \
    --log_suffix "$TRAIN_SUFFIX" \
    --log_root "$TRAIN_RUNS_ROOT"

  LAST_TRAIN_RUN=$(ls -1dt "$TRAIN_RUNS_ROOT"/*-train-"$TRAIN_SUFFIX" 2>/dev/null | head -n 1 || true)
  if [[ -z "$LAST_TRAIN_RUN" ]]; then
    echo "Could not find train run under $TRAIN_RUNS_ROOT"
    exit 1
  fi

  MODEL_PATH=$(ls -1t "$LAST_TRAIN_RUN"/checkpoints/sl_checkpoint_*_199.pth 2>/dev/null | head -n 1 || true)
  if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH=$(ls -1t "$LAST_TRAIN_RUN"/checkpoints/sl_checkpoint_*.pth 2>/dev/null | head -n 1 || true)
  fi
  if [[ -z "$MODEL_PATH" ]]; then
    echo "Could not find checkpoint in $LAST_TRAIN_RUN/checkpoints"
    exit 1
  fi

  echo "$MODEL_PATH" > "$EXP_ROOT/model_path.txt"
  echo "[train] checkpoint=$MODEL_PATH"
fi

if [[ "$MODE" == "benchmark" || "$MODE" == "all" ]]; then
  if [[ -z "$MODEL_PATH" ]]; then
    if [[ -f "$EXP_ROOT/model_path.txt" ]]; then
      MODEL_PATH=$(cat "$EXP_ROOT/model_path.txt")
    fi
  fi

  if [[ -z "$MODEL_PATH" ]]; then
    echo "No model path provided. Use --model-path or run with --mode train/all first."
    exit 1
  fi
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model checkpoint not found: $MODEL_PATH"
    exit 1
  fi

  BENCH_SUFFIX="${NAME}-bench-$(date +%Y%m%d-%H%M%S)"
  echo "[benchmark] name=$NAME suffix=$BENCH_SUFFIX gpu=$GPU parallel=$PARALLEL"
  "$PYTHON_BIN" "$ROOT/scripts/test/run_all_benchmarks.py" \
    --model-path "$MODEL_PATH" \
    --log-suffix "$BENCH_SUFFIX" \
    --gpu "$GPU" \
    --parallel "$PARALLEL" \
    --progress 1 \
    --runs-root "$BENCH_RUNS_ROOT" \
    --console-dir "$BENCH_CONSOLE_DIR" \
    --python-bin "$PYTHON_BIN"

  echo "[benchmark] summaries:"
  "$PYTHON_BIN" "$ROOT/scripts/test/summarize_logs.py" --runs-root "$BENCH_RUNS_ROOT"
fi

echo "[done] experiment_root=$EXP_ROOT"
