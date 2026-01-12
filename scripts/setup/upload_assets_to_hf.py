#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _resolve_token(arg_token: str | None) -> str | None:
    if arg_token:
        return arg_token
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        from huggingface_hub.utils import get_token

        token = get_token()
        if token:
            return token
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass
    return None


def _validate_dir(path: Path, label: str) -> bool:
    if not path.exists():
        print(f"error: {label} not found at {path}", file=sys.stderr)
        return False
    if not path.is_dir():
        print(f"error: {label} is not a directory: {path}", file=sys.stderr)
        return False
    return True


def _scan_upload_cache(cache_root: Path) -> tuple[int, int, int, float | None]:
    if not cache_root.exists():
        return 0, 0, 0, None

    total = 0
    committed = 0
    uploaded = 0
    last_ts: float | None = None

    for meta_path in cache_root.rglob("*.metadata"):
        total += 1
        try:
            lines = meta_path.read_text(errors="ignore").splitlines()
        except OSError:
            continue

        while len(lines) < 8:
            lines.append("")

        try:
            ts = float(lines[0].strip())
        except ValueError:
            ts = None
        if ts is not None and (last_ts is None or ts > last_ts):
            last_ts = ts

        if lines[6].strip() == "1":
            uploaded += 1
        if lines[7].strip() == "1":
            committed += 1

    return total, committed, uploaded, last_ts


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    return f"{int(seconds // 3600)}h"


def _start_watch_thread(cache_root: Path, interval: int, stop_event: threading.Event) -> threading.Thread:
    def _watch() -> None:
        last_report: tuple[int, int, int] | None = None
        while not stop_event.is_set():
            total, committed, uploaded, last_ts = _scan_upload_cache(cache_root)
            if total == 0:
                print(f"[watch] waiting for metadata in {cache_root}")
            else:
                pending = total - committed
                changed = last_report != (total, committed, uploaded)
                if last_ts is not None:
                    age = _format_age(time.time() - last_ts)
                    suffix = f"last update {age} ago"
                else:
                    suffix = "last update unknown"
                line = f"[watch] committed {committed}/{total}, uploaded {uploaded}/{total}, pending {pending} ({suffix})"
                if changed:
                    print(line)
                else:
                    print(line)
                last_report = (total, committed, uploaded)
            stop_event.wait(interval)

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()
    return thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Hugging Face dataset repo and upload A2 asset folders."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id (e.g., username/a2_assets).",
    )
    parser.add_argument(
        "--assets-root",
        default=str(ROOT / "assets"),
        help="Path to local assets directory (default: ./assets).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public dataset (default: private).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--commit-message",
        default="Add A2 assets",
        help="Base commit message for uploads.",
    )
    parser.add_argument(
        "--no-large",
        action="store_true",
        help="Use upload_folder (not resumable for large uploads).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers for large upload (default: hub chooses).",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=60,
        help="Seconds between progress reports for large upload.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Print local cache progress while uploading (helps detect stalls).",
    )
    parser.add_argument(
        "--watch-interval",
        type=int,
        default=60,
        help="Seconds between local progress checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_root = Path(args.assets_root).resolve()
    simplified_dir = assets_root / "simplified_objects"
    unseen_dir = assets_root / "unseen_objects"

    if not _validate_dir(assets_root, "assets root"):
        return 1
    if not _validate_dir(simplified_dir, "simplified_objects"):
        return 1
    if not _validate_dir(unseen_dir, "unseen_objects"):
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed; run `uv sync`.", file=sys.stderr)
        return 1

    token = _resolve_token(args.token)
    if not token:
        print(
            "error: no Hugging Face token found. Set HF_TOKEN or run `huggingface-cli login`.",
            file=sys.stderr,
        )
        return 1

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    use_large = hasattr(api, "upload_large_folder") and not args.no_large
    if use_large:
        try:
            from huggingface_hub import constants
            from huggingface_hub.utils._runtime import is_xet_available
        except Exception:
            constants = None
            is_xet_available = None

        if constants is not None and constants.HF_HUB_DISABLE_XET:
            print(
                "error: HF_HUB_DISABLE_XET is set. Unset it to enable Xet uploads.",
                file=sys.stderr,
            )
            return 1
        if is_xet_available is None or not is_xet_available():
            print(
                "error: hf_xet is required for binary asset uploads. Install it and rerun:\n"
                "  uv pip install hf_xet",
                file=sys.stderr,
            )
            return 1

        print("Using upload_large_folder for resumable uploads (creates .cache/.huggingface).")
        stop_event = threading.Event()
        watch_thread: threading.Thread | None = None
        if args.watch:
            cache_root = assets_root / ".cache" / "huggingface" / "upload"
            watch_thread = _start_watch_thread(cache_root, args.watch_interval, stop_event)
        ignore_patterns = ["**/.cache/**", ".cache/**", "**/.huggingface/**"]
        allow_patterns = ["simplified_objects/**", "unseen_objects/**"]
        try:
            api.upload_large_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=str(assets_root),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                num_workers=args.num_workers,
                print_report=True,
                print_report_every=args.report_every,
            )
        finally:
            stop_event.set()
            if watch_thread is not None:
                watch_thread.join(timeout=5)
    else:
        print(f"Uploading {simplified_dir} -> {args.repo_id}:simplified_objects")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(simplified_dir),
            path_in_repo="simplified_objects",
            commit_message=f"{args.commit_message} (simplified_objects)",
        )

        print(f"Uploading {unseen_dir} -> {args.repo_id}:unseen_objects")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(unseen_dir),
            path_in_repo="unseen_objects",
            commit_message=f"{args.commit_message} (unseen_objects)",
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
