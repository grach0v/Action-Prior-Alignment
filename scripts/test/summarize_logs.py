#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import math


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
AVG_SUCCESS_RE = re.compile(r"average success:\s*([0-9.]+)", re.IGNORECASE)
LANG_GOAL_RE = re.compile(r"Language goal:\s*(.+)")
RESET_GOAL_RE = re.compile(r"Reset environment of episode\s+\d+,\s*(.+)")
SUCCESS_RE = re.compile(r"success:\s*(True|False)")
SCI_FLOAT_RE = re.compile(r"[-+]?\d+\.\d+e[-+]\d+", re.IGNORECASE)

RUN_NAME_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-(?P<rest>.+)$")
TASK_RE = re.compile(r"(?:^|-)test-(?P<task>grasp|place|pp)(?:-|$)")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def _parse_floats(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        if math.isfinite(v):
            values.append(v)
    return values


def _parse_run_name(run_name: str) -> tuple[str | None, str | None, str | None]:
    """Return (date, task, split) from a run directory name."""
    m = RUN_NAME_RE.match(run_name)
    date = m.group("date") if m else None

    task = None
    m2 = TASK_RE.search(run_name)
    if m2:
        task = m2.group("task")
        if task == "pp":
            task = "pickplace"

    split = "unseen" if "unseen" in run_name else "seen"
    return date, task, split


def extract_summary(log_path):
    text = strip_ansi(log_path.read_text(errors="ignore"))
    avg_rows = []
    reset_goals = []
    success_values = []

    for line in text.splitlines():
        match = LANG_GOAL_RE.search(line)
        if match:
            goal = match.group(1).strip()
            avg = AVG_SUCCESS_RE.search(line)
            if avg:
                avg_rows.append((goal, float(avg.group(1))))
            continue

        match = RESET_GOAL_RE.search(line)
        if match:
            reset_goals.append(match.group(1).strip())
            continue

        match = SUCCESS_RE.search(line)
        if match:
            success_values.append(match.group(1) == "True")

    if avg_rows:
        return avg_rows

    if success_values:
        avg = sum(1 for v in success_values if v) / len(success_values)
        goal = reset_goals[0] if reset_goals else "-"
        return [(goal, avg)]

    if reset_goals:
        return [(reset_goals[0], None)]

    return []


def read_success_log(success_path):
    values = _parse_floats(success_path)
    if not values:
        return None
    return sum(values) / len(values)


def _parse_case_metrics(case_path: Path):
    """Parse one case result file written by evaluator.save_case_results()."""
    for line in case_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        values = [float(tok) for tok in SCI_FLOAT_RE.findall(line)]
        # grasp/place (old): [success, avg_step, avg_success_step, avg_reward]
        if len(values) == 4:
            return {
                "success_strict": values[0],
                "success_skip": values[0],
                "avg_success_step": values[2],
                "strict_weight": 1.0,
                "skip_weight": 1.0,
            }
        # pickplace (old): [success, grasp_success, place_success, avg_step, avg_success_step]
        if len(values) == 5:
            return {
                "success_strict": values[0],
                "success_skip": values[0],
                "avg_success_step": values[4],
                "strict_weight": 1.0,
                "skip_weight": 1.0,
            }
        # pickplace (new): old fields + [skip_success, valid_episodes, total_episodes]
        if len(values) >= 8:
            return {
                "success_strict": values[0],
                "success_skip": values[5],
                "avg_success_step": values[4],
                "skip_weight": values[6],
                "strict_weight": values[7],
            }
        # grasp/place (new): old fields + [skip_success, valid_episodes, total_episodes]
        if len(values) >= 7:
            return {
                "success_strict": values[0],
                "success_skip": values[4],
                "avg_success_step": values[2],
                "skip_weight": values[5],
                "strict_weight": values[6],
            }
    return None


def _summarize_from_results(run_dir: Path):
    results_dir = run_dir / "results"
    if not results_dir.exists():
        return None

    rows = []
    for case_path in sorted(results_dir.glob("case*.txt")):
        case_metrics = _parse_case_metrics(case_path)
        if case_metrics is not None:
            rows.append(case_metrics)

    if not rows:
        return None

    strict_den = sum(max(0.0, r.get("strict_weight", 1.0)) for r in rows)
    skip_den = sum(max(0.0, r.get("skip_weight", 1.0)) for r in rows)
    strict_num = sum(
        r["success_strict"] * max(0.0, r.get("strict_weight", 1.0)) for r in rows
    )
    skip_num = sum(
        r["success_skip"] * max(0.0, r.get("skip_weight", 1.0)) for r in rows
    )
    success_rate_strict = strict_num / strict_den if strict_den > 0 else None
    success_rate_skip = skip_num / skip_den if skip_den > 0 else None
    avg_success_step = sum(r["avg_success_step"] for r in rows) / len(rows)
    return {
        "success_rate_strict": success_rate_strict,
        "success_rate_skip": success_rate_skip,
        "avg_success_step": avg_success_step,
        "n_cases": len(rows),
        "n_strict_episodes": strict_den,
        "n_skip_episodes": skip_den,
        "source": "results",
    }


def _summarize_from_transitions(run_dir: Path):
    """Fallback for old/incomplete runs that do not have result files."""
    success_path = run_dir / "transitions" / "episode_success.log.txt"
    step_path = run_dir / "transitions" / "episode_step.log.txt"
    if not success_path.exists() or not step_path.exists():
        return None

    success = _parse_floats(success_path)
    steps = _parse_floats(step_path)
    if not success or not steps or len(success) != len(steps):
        return None

    success_rate = sum(1.0 for v in success if v > 0.5) / len(success)
    success_steps = [s for s, ok in zip(steps, success, strict=True) if ok > 0.5]
    avg_success_step = sum(success_steps) / len(success_steps) if success_steps else None
    return {
        "success_rate_strict": success_rate,
        "success_rate_skip": success_rate,
        "avg_success_step": avg_success_step,
        "n_cases": len(success),
        "n_strict_episodes": len(success),
        "n_skip_episodes": len(success),
        "source": "transitions_last_case",
    }


def summarize_run_dir(run_dir):
    run_dir = Path(run_dir)

    date, task, split = _parse_run_name(run_dir.name)
    if date is None:
        return None
    if task is None:
        # Fallback: infer from which transition files exist.
        if (run_dir / "transitions" / "episode_place_success.log.txt").exists() or (
            run_dir / "transitions" / "episode_grasp_success.log.txt"
        ).exists():
            task = "pickplace"
        else:
            task = "-"

    metrics = _summarize_from_results(run_dir)
    if metrics is None:
        metrics = _summarize_from_transitions(run_dir)
    if metrics is None:
        return None

    return {
        "date": date,
        "task": task,
        "split": split,
        "success_rate_strict": metrics["success_rate_strict"],
        "success_rate_skip": metrics["success_rate_skip"],
        "avg_success_step": metrics["avg_success_step"],
        "n_cases": metrics["n_cases"],
        "n_strict_episodes": metrics.get("n_strict_episodes"),
        "n_skip_episodes": metrics.get("n_skip_episodes"),
        "source": metrics["source"],
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark log files")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument(
        "--pattern",
        default="test_*.out,run_*.out",
        help="Comma-separated glob patterns",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Single run directory under logs (with transitions)",
    )
    parser.add_argument(
        "--runs-root",
        default="",
        help="Scan all run directories under this root",
    )
    args = parser.parse_args()

    if args.run_dir:
        result = summarize_run_dir(args.run_dir)
        if result is None:
            raise SystemExit(f"No valid benchmark summary in {args.run_dir}")
        avg_step = f"{result['avg_success_step']:.3f}" if result["avg_success_step"] is not None else "-"
        strict = f"{result['success_rate_strict']:.4f}" if result["success_rate_strict"] is not None else "-"
        skip = f"{result['success_rate_skip']:.4f}" if result["success_rate_skip"] is not None else "-"
        print("date\ttask\tsplit\tsuccess_rate_strict\tsuccess_rate_skip\tavg_success_step\tn_cases\tsource")
        print(
            f"{result['date']}\t{result['task']}\t{result['split']}\t"
            f"{strict}\t{skip}\t"
            f"{avg_step}\t{result['n_cases']}\t{result['source']}"
        )
        return

    if args.runs_root:
        root = Path(args.runs_root)
        if not root.exists():
            raise SystemExit(f"Runs root not found: {root}")
        print("date\ttask\tsplit\tsuccess_rate_strict\tsuccess_rate_skip\tavg_success_step\tn_cases\tsource")
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            result = summarize_run_dir(run_dir)
            if result is None:
                continue
            avg_step = f"{result['avg_success_step']:.3f}" if result["avg_success_step"] is not None else "-"
            strict = f"{result['success_rate_strict']:.4f}" if result["success_rate_strict"] is not None else "-"
            skip = f"{result['success_rate_skip']:.4f}" if result["success_rate_skip"] is not None else "-"
            print(
                f"{result['date']}\t{result['task']}\t{result['split']}\t"
                f"{strict}\t{skip}\t"
                f"{avg_step}\t{result['n_cases']}\t{result['source']}"
            )
        return

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    patterns = [p.strip() for p in args.pattern.split(",") if p.strip()]
    logs = []
    for pattern in patterns:
        logs.extend(log_dir.glob(pattern))
    logs = sorted({p for p in logs if p.name != "summarize_logs.out"})
    if not logs:
        raise SystemExit(f"No .out logs found in {log_dir}")

    print("log_file\ttask\taverage_success")
    for log_path in logs:
        rows = extract_summary(log_path)
        for task, avg_success in rows:
            success = f"{avg_success:.4f}" if avg_success is not None else "-"
            print(f"{log_path.name}\t{task}\t{success}")


if __name__ == "__main__":
    main()
