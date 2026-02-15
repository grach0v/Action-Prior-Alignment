import argparse
import os
import re


NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def infer_task_and_split(session_dir):
    name = os.path.basename(session_dir)
    split = "unseen" if "-unseen-" in name else "seen"
    if "-test-grasp-" in name:
        task = "pick"
    elif "-test-place-" in name:
        task = "place"
    elif "-test-pp-" in name:
        task = "pp"
    else:
        task = None
    return task, split


def infer_case_dir(task, split):
    if task == "pick":
        return f"testing_cases/grasp_testing_cases/{split}"
    if task == "place":
        return f"testing_cases/place_testing_cases/{split}"
    if task == "pp":
        return f"testing_cases/pp_testing_cases/{split}"
    return None


def get_case_names(case_dir):
    if not case_dir or not os.path.isdir(case_dir):
        return []
    names = sorted(os.listdir(case_dir), key=lambda x: int(x[4:6]))
    return names


def parse_result_file(path, task):
    with open(path, "r") as f:
        line = f.readline().strip()
    nums = [float(x) for x in NUM_RE.findall(line)]
    if task in {"pick", "place"} and len(nums) >= 4:
        return {
            "success": nums[-4],
            "avg_step": nums[-3],
            "avg_success_step": nums[-2],
            "avg_reward": nums[-1],
            "raw": line,
        }
    if task == "pp" and len(nums) >= 5:
        return {
            "success": nums[-5],
            "grasp_success": nums[-4],
            "place_success": nums[-3],
            "avg_step": nums[-2],
            "avg_success_step": nums[-1],
            "raw": line,
        }
    return {"raw": line}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dir", required=True)
    parser.add_argument("--task", choices=["pick", "place", "pp"], default=None)
    parser.add_argument("--case_dir", default=None)
    parser.add_argument("--success_threshold", type=float, default=0.9999)
    parser.add_argument("--print_failed_case_files", action="store_true", default=False)
    args = parser.parse_args()

    inferred_task, split = infer_task_and_split(args.session_dir)
    task = args.task or inferred_task
    if task is None:
        raise ValueError("Cannot infer task from session dir name; pass --task")

    case_dir = args.case_dir or infer_case_dir(task, split)
    case_names = get_case_names(case_dir)

    results_dir = os.path.join(args.session_dir, "results")
    files = sorted(
        [x for x in os.listdir(results_dir) if x.startswith("case") and x.endswith(".txt")],
        key=lambda x: int(x[4:-4]),
    )

    rows = []
    for name in files:
        case_idx = int(name[4:-4])
        case_name = case_names[case_idx] if case_idx < len(case_names) else "unknown"
        path = os.path.join(results_dir, name)
        parsed = parse_result_file(path, task)
        parsed["case_idx"] = case_idx
        parsed["case_file"] = case_name
        rows.append(parsed)

    failed = [r for r in rows if r.get("success", 0.0) < args.success_threshold]
    failed = sorted(failed, key=lambda x: x.get("success", 0.0))

    if args.print_failed_case_files:
        for r in failed:
            print(r["case_file"])
        return

    print(f"Session: {args.session_dir}")
    print(f"Task: {task}, Split: {split}")
    print(f"Total cases: {len(rows)}")
    print(f"Failed cases (success < {args.success_threshold}): {len(failed)}")
    print("")

    for r in failed:
        if task == "pp":
            print(
                f"case{r['case_idx']:02d} ({r['case_file']}): "
                f"success={r.get('success', 0.0):.3f}, "
                f"grasp={r.get('grasp_success', 0.0):.3f}, "
                f"place={r.get('place_success', 0.0):.3f}, "
                f"avg_step={r.get('avg_step', 0.0):.2f}"
            )
        else:
            print(
                f"case{r['case_idx']:02d} ({r['case_file']}): "
                f"success={r.get('success', 0.0):.3f}, "
                f"avg_step={r.get('avg_step', 0.0):.2f}, "
                f"avg_success_step={r.get('avg_success_step', 0.0):.2f}"
            )


if __name__ == "__main__":
    main()
