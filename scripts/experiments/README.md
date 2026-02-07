# Experiment Workflow

Use one script for unified training + benchmark with organized logs:

```bash
bash scripts/experiments/train_and_benchmark_unified.sh --name my_exp --gpu 0 --mode all
```

Benchmark only with an existing checkpoint:

```bash
bash scripts/experiments/train_and_benchmark_unified.sh \
  --name my_exp \
  --mode benchmark \
  --model-path /absolute/path/to/checkpoint.pth
```

Outputs are grouped under:

```text
logs/experiments/<name>/
  train/runs/
  benchmark/runs/
  benchmark/console/
  model_path.txt
```
