#!/usr/bin/env python3

import argparse
import glob
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


EPISODE_RE = re.compile(r"Episode:\s*(\d+)(?:/(\d+))?,")
SPLIT_RE = re.compile(r"^(seen|unseen)$")
RESET_EVENT_RE = re.compile(r"Reset environment of episode\s+(\d+),")
RESET_FAILED_RE = re.compile(r"Reset failed for episode\s+\d+\s+after\s+\d+\s+retries")


@dataclass
class TaskSpec:
    name: str
    script: str
    gpu: str
    run_pattern_seen: str
    run_pattern_unseen: str
    case_dir_seen: str
    case_dir_unseen: str
    log_path: Optional[Path] = None
    proc: Optional[subprocess.Popen] = None
    split: str = "-"
    episode_done: int = 0
    episode_total: Optional[int] = None
    seen_done: int = 0
    unseen_done: int = 0
    reset_events: int = 0
    reset_failed: int = 0
    done: bool = False


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, msg: str) -> None:
        with self._lock:
            print(msg, flush=True)
            self._fp.write(msg + "\n")
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            self._fp.close()


def count_expected_cases(case_dir: Path) -> int:
    if not case_dir.exists():
        return 0
    return len(list(case_dir.glob("case*.txt")))


def count_result_files(run_dir: Optional[Path]) -> int:
    if run_dir is None:
        return 0
    results_dir = run_dir / "results"
    if not results_dir.exists():
        return 0
    return len(list(results_dir.glob("case*.txt")))


def latest_run_dir(pattern: str) -> Optional[Path]:
    matches = [Path(p) for p in glob.glob(pattern)]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def read_tail(path: Path, n: int = 120) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(errors="ignore").splitlines()
    return lines[-n:]


def parse_run_stamp(name: str) -> Optional[datetime]:
    parts = name.split("-")
    if len(parts) < 6:
        return None
    try:
        return datetime.strptime("-".join(parts[:6]), "%Y-%m-%d-%H-%M-%S")
    except Exception:
        return None


def latest_verified_run(root: Path, pattern: str) -> Optional[Path]:
    matches: List[Tuple[datetime, Path]] = []
    for path in glob.glob(str(root / pattern)):
        p = Path(path)
        if not p.is_dir():
            continue
        ts = parse_run_stamp(p.name)
        if ts is None:
            continue
        matches.append((ts, p))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def expected_case_set(case_dir: Path) -> Set[str]:
    out: Set[str] = set()
    for name in os.listdir(case_dir):
        if not name.endswith(".txt"):
            continue
        m = re.match(r"case(\d+)", name)
        if not m:
            continue
        out.add(f"case{int(m.group(1))}.txt")
    return out


def result_case_set(run_dir: Path) -> Set[str]:
    out: Set[str] = set()
    results_dir = run_dir / "results"
    if not results_dir.exists():
        return out
    for name in os.listdir(results_dir):
        if name.startswith("case") and name.endswith(".txt"):
            out.add(name)
    return out


def case_progress(split: str, seen_done: int, unseen_done: int, seen_total: int, unseen_total: int) -> str:
    if split == "seen":
        if seen_total <= 0:
            return "0/?"
        return f"{min(seen_done + 1, seen_total)}/{seen_total}"
    if split == "unseen":
        if unseen_total <= 0:
            return "0/?"
        return f"{min(unseen_done + 1, unseen_total)}/{unseen_total}"
    return "0/?"


def episode_progress(done: int, total: Optional[int]) -> str:
    if total is None:
        return f"{done}/?"
    return f"{done}/{total}"


def progress_line(tasks: List[TaskSpec], expected: Dict[str, Tuple[int, int]]) -> str:
    parts: List[str] = []
    for task in tasks:
        seen_total, unseen_total = expected[task.name]
        extras: List[str] = []
        if task.reset_events > 0:
            extras.append(f"reset_events {task.reset_events}")
        if task.reset_failed > 0:
            extras.append(f"reset_failed {task.reset_failed}")
        extra_txt = f" ({', '.join(extras)})" if extras else ""
        parts.append(
            f"{task.name} {task.split} case {case_progress(task.split, task.seen_done, task.unseen_done, seen_total, unseen_total)} "
            f"episode {episode_progress(task.episode_done, task.episode_total)}{extra_txt}"
        )
    return " | ".join(parts)


def refresh_case_counts(runs_root: Path, suffix: str, task: TaskSpec) -> None:
    run_seen = latest_run_dir(str(runs_root / task.run_pattern_seen.format(suffix=suffix)))
    run_unseen = latest_run_dir(str(runs_root / task.run_pattern_unseen.format(suffix=suffix)))
    task.seen_done = count_result_files(run_seen)
    task.unseen_done = count_result_files(run_unseen)


def parse_task_line(runs_root: Path, suffix: str, task: TaskSpec, line: str) -> bool:
    changed = False
    stripped = line.strip()

    split_match = SPLIT_RE.match(stripped)
    if split_match:
        split = split_match.group(1)
        if split != task.split:
            task.split = split
            changed = True

    ep_match = EPISODE_RE.search(line)
    if ep_match:
        raw_done = int(ep_match.group(1))
        raw_total = ep_match.group(2)
        if raw_total is None:
            done = raw_done + 1
            total = None
        else:
            done = raw_done
            total = int(raw_total)
        if done != task.episode_done or total != task.episode_total:
            task.episode_done = done
            task.episode_total = total
            changed = True

    reset_match = RESET_EVENT_RE.search(line)
    if reset_match:
        task.reset_events += 1
        # Emit progress periodically even if episode/case counters are unchanged.
        if task.reset_events % 10 == 0:
            changed = True

    if RESET_FAILED_RE.search(line):
        task.reset_failed += 1
        changed = True

    if changed or "average success:" in line or "Reset environment of episode" in line:
        prev_seen, prev_unseen = task.seen_done, task.unseen_done
        refresh_case_counts(runs_root, suffix, task)
        if task.seen_done != prev_seen or task.unseen_done != prev_unseen:
            changed = True

    return changed


def spawn_task(
    root: Path,
    logger: Logger,
    task: TaskSpec,
    suffix: str,
    model_path: str,
    runs_root: Path,
    console_dir: Path,
    python_bin: str,
    event_q: "queue.Queue[Tuple[str, str, str]]",
) -> None:
    task_log = console_dir / f"{task.name}-{suffix}.log"
    task.log_path = task_log
    task_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash",
        str(root / task.script),
        "--gpu",
        task.gpu,
        "--model-path",
        model_path,
        "--log-suffix",
        suffix,
        "--log-root",
        str(runs_root),
        "--python-bin",
        python_bin,
    ]
    logger.log(f"[start] {task.name} gpu={task.gpu} log={task_log}")
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    task.proc = proc

    def _reader() -> None:
        assert proc.stdout is not None
        with task_log.open("w", encoding="utf-8") as fp:
            for raw in proc.stdout:
                fp.write(raw)
                fp.flush()
                event_q.put(("line", task.name, raw.rstrip("\n")))
        proc.stdout.close()

    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def run_group(
    root: Path,
    logger: Logger,
    tasks: List[TaskSpec],
    suffix: str,
    model_path: str,
    runs_root: Path,
    console_dir: Path,
    python_bin: str,
    expected: Dict[str, Tuple[int, int]],
    progress: bool,
) -> int:
    event_q: "queue.Queue[Tuple[str, str, str]]" = queue.Queue()
    start_ts = time.time()
    last_progress = ""
    by_name = {t.name: t for t in tasks}

    for task in tasks:
        spawn_task(root, logger, task, suffix, model_path, runs_root, console_dir, python_bin, event_q)

    if progress:
        logger.log("[progress] mode=event (stdout lines)")
        for task in tasks:
            refresh_case_counts(runs_root, suffix, task)
        line = progress_line(tasks, expected)
        logger.log(f"[progress t=0s] {line}")
        last_progress = line

    alive = set(t.name for t in tasks)
    failed = False
    while alive:
        try:
            _, name, line = event_q.get(timeout=0.2)
        except queue.Empty:
            line = ""
            name = ""

        if name:
            task = by_name[name]
            changed = parse_task_line(runs_root, suffix, task, line)
            if changed and progress:
                now = int(time.time() - start_ts)
                current = progress_line(tasks, expected)
                if current != last_progress:
                    logger.log(f"[progress t={now}s] {current}")
                    last_progress = current

        for task in tasks:
            if task.name not in alive:
                continue
            assert task.proc is not None
            rc = task.proc.poll()
            if rc is None:
                continue
            alive.remove(task.name)
            task.done = True
            if rc == 0:
                logger.log(f"[done] {task.name}")
            else:
                failed = True
                logger.log(f"[fail] {task.name} (log: {task.log_path})")
                for tail_line in read_tail(task.log_path or Path("/dev/null"), 120):
                    logger.log(tail_line)

    if progress:
        for task in tasks:
            refresh_case_counts(runs_root, suffix, task)
        now = int(time.time() - start_ts)
        current = progress_line(tasks, expected)
        if current != last_progress:
            logger.log(f"[progress t={now}s] {current}")

    return 1 if failed else 0


def verify_outputs(root: Path, runs_root: Path, logger: Logger, suffix: str) -> bool:
    checks = {
        "grasp-seen": {
            "run_pattern": f"*test-grasp-{suffix}",
            "case_dir": root / "testing_cases" / "grasp_testing_cases" / "seen",
        },
        "grasp-unseen": {
            "run_pattern": f"*test-grasp-{suffix}-unseen",
            "case_dir": root / "testing_cases" / "grasp_testing_cases" / "unseen",
        },
        "place-seen": {
            "run_pattern": f"*test-place-{suffix}",
            "case_dir": root / "testing_cases" / "place_testing_cases" / "seen",
        },
        "place-unseen": {
            "run_pattern": f"*test-place-{suffix}-unseen",
            "case_dir": root / "testing_cases" / "place_testing_cases" / "unseen",
        },
        "pickplace-seen": {
            "run_pattern": f"*test-pp-{suffix}",
            "case_dir": root / "testing_cases" / "pp_testing_cases" / "seen",
        },
        "pickplace-unseen": {
            "run_pattern": f"*test-pp-{suffix}-unseen",
            "case_dir": root / "testing_cases" / "pp_testing_cases" / "unseen",
        },
    }

    failed = False
    logger.log(f"verification suffix={suffix}")
    for key, cfg in checks.items():
        run_dir = latest_verified_run(runs_root, cfg["run_pattern"])
        if run_dir is None:
            logger.log(f"{key}: MISSING run dir for pattern {cfg['run_pattern']}")
            failed = True
            continue
        expected = expected_case_set(cfg["case_dir"])
        got = result_case_set(run_dir)
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        status = "OK" if not missing and not extra else "MISMATCH"
        logger.log(f"{key}: {status} results={len(got)} expected={len(expected)} dir={run_dir}")
        if missing:
            logger.log(f"  missing={','.join(missing)}")
        if extra:
            logger.log(f"  extra={','.join(extra)}")
        if missing or extra:
            failed = True
    return not failed


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run all benchmark tasks and verify outputs.")
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "a2_pretrained/checkpoints/sl_checkpoint_199.pth"),
    )
    parser.add_argument("--log-suffix", default=os.environ.get("LOG_SUFFIX", f"bench-{time.strftime('%Y%m%d-%H%M%S')}"))
    parser.add_argument("--runs-root", default=os.environ.get("RUNS_ROOT", str(root / "logs")))
    parser.add_argument("--console-dir", default=os.environ.get("LOG_DIR", str(root / "logs")))
    parser.add_argument("--log-file", default=os.environ.get("LOG_FILE", ""))
    parser.add_argument("--parallel", type=int, choices=[0, 1], default=int(os.environ.get("PARALLEL", "1")))
    parser.add_argument("--progress", type=int, choices=[0, 1], default=int(os.environ.get("PROGRESS", "1")))
    parser.add_argument("--gpu-grasp", default=os.environ.get("GPU_GRASP"))
    parser.add_argument("--gpu-place", default=os.environ.get("GPU_PLACE"))
    parser.add_argument("--gpu-pickplace", default=os.environ.get("GPU_PICKPLACE"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", str(root / ".venv" / "bin" / "python")))
    args = parser.parse_args()

    gpu = str(args.gpu)
    model_path = args.model_path
    log_suffix = args.log_suffix
    runs_root = Path(args.runs_root)
    console_dir = Path(args.console_dir)
    parallel = args.parallel == 1
    progress = args.progress == 1

    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = console_dir / f"run_all_benchmarks-{log_suffix}.log"

    tasks = [
        TaskSpec(
            name="grasp",
            script="scripts/test/test_grasp.sh",
            gpu=str(args.gpu_grasp) if args.gpu_grasp is not None else gpu,
            run_pattern_seen="*-test-grasp-{suffix}",
            run_pattern_unseen="*-test-grasp-{suffix}-unseen",
            case_dir_seen="testing_cases/grasp_testing_cases/seen",
            case_dir_unseen="testing_cases/grasp_testing_cases/unseen",
        ),
        TaskSpec(
            name="place",
            script="scripts/test/test_place.sh",
            gpu=str(args.gpu_place) if args.gpu_place is not None else gpu,
            run_pattern_seen="*-test-place-{suffix}",
            run_pattern_unseen="*-test-place-{suffix}-unseen",
            case_dir_seen="testing_cases/place_testing_cases/seen",
            case_dir_unseen="testing_cases/place_testing_cases/unseen",
        ),
        TaskSpec(
            name="pickplace",
            script="scripts/test/test_pickplace.sh",
            gpu=str(args.gpu_pickplace) if args.gpu_pickplace is not None else gpu,
            run_pattern_seen="*-test-pp-{suffix}",
            run_pattern_unseen="*-test-pp-{suffix}-unseen",
            case_dir_seen="testing_cases/pp_testing_cases/seen",
            case_dir_unseen="testing_cases/pp_testing_cases/unseen",
        ),
    ]

    expected: Dict[str, Tuple[int, int]] = {
        t.name: (
            count_expected_cases(root / t.case_dir_seen),
            count_expected_cases(root / t.case_dir_unseen),
        )
        for t in tasks
    }

    logger = Logger(log_file)
    try:
        logger.log(f"MODEL_PATH={model_path}")
        logger.log(f"LOG_SUFFIX={log_suffix}")
        logger.log(f"PARALLEL={1 if parallel else 0}")
        logger.log(f"GPU_GRASP={tasks[0].gpu}")
        logger.log(f"GPU_PLACE={tasks[1].gpu}")
        logger.log(f"GPU_PICKPLACE={tasks[2].gpu}")
        logger.log(f"RUNS_ROOT={runs_root}")
        logger.log(f"CONSOLE_DIR={console_dir}")
        logger.log(f"PYTHON_BIN={args.python_bin}")
        logger.log(f"LOG_FILE={log_file}")
        logger.log(f"PROGRESS={1 if progress else 0}")

        rc = 0
        if parallel:
            rc = run_group(root, logger, tasks, log_suffix, model_path, runs_root, console_dir, args.python_bin, expected, progress)
        else:
            for task in tasks:
                rc = run_group(
                    root,
                    logger,
                    [task],
                    log_suffix,
                    model_path,
                    runs_root,
                    console_dir,
                    args.python_bin,
                    expected,
                    progress,
                )
                if rc != 0:
                    break

        if rc != 0:
            return rc

        ok = verify_outputs(root, runs_root, logger, log_suffix)
        if not ok:
            logger.log("Benchmark verification failed: missing/extra case outputs.")
            return 1
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
