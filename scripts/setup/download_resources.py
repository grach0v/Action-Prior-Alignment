#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ASSETS_URL = "https://drive.google.com/drive/folders/1WxKDFXJktoqiP0jmkDZrMCcNNBx5u-YM?usp=drive_link"
TESTING_CASES_URL = "https://drive.google.com/drive/folders/1OuTua-69NEeV7RYIi9nzR1jmdZEugB68?usp=sharing"
PRETRAINED_URL = "https://drive.google.com/drive/folders/1uoDGIgkcSi8okcr8qjKOaF57TyRaHRd_?usp=sharing"
HF_DATASET_REPO = "KechunXu1/A2_Dataset"


def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _archive_kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar"):
        return "tar"
    return None


def _safe_extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = dest / member.filename
            if not _is_within_directory(dest, target):
                raise RuntimeError(f"unsafe path in zip: {member.filename}")
        zf.extractall(dest)


def _safe_extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            target = dest / member.name
            if not _is_within_directory(dest, target):
                raise RuntimeError(f"unsafe path in tar: {member.name}")
        tf.extractall(dest)


def _find_archives(target_dir: Path) -> list[Path]:
    candidates = list(target_dir.iterdir())
    archives = [p for p in candidates if p.is_file() and _archive_kind(p)]
    if archives:
        return archives

    subdirs = [p for p in candidates if p.is_dir()]
    if len(subdirs) == 1:
        return [p for p in subdirs[0].iterdir() if p.is_file() and _archive_kind(p)]

    return []


def extract_archives(target_dir: Path, cleanup: bool) -> None:
    archives = _find_archives(target_dir)
    if not archives:
        print(f"No archives found in {target_dir}")
        return

    for archive in archives:
        kind = _archive_kind(archive)
        print(f"Extracting {archive.name} -> {target_dir}")
        if kind == "zip":
            _safe_extract_zip(archive, target_dir)
        elif kind == "tar":
            _safe_extract_tar(archive, target_dir)
        if cleanup:
            archive.unlink()


def _normalize_download_root(target_dir: Path, before: set[str]) -> None:
    # If the download created a single nested folder, lift its contents up.
    after = [p for p in target_dir.iterdir() if p.name not in before]
    new_files = [p for p in after if p.is_file()]
    new_dirs = [p for p in after if p.is_dir()]
    if new_files or len(new_dirs) != 1:
        return

    nested = new_dirs[0]
    for child in nested.iterdir():
        dest = target_dir / child.name
        if dest.exists():
            continue
        shutil.move(str(child), str(dest))
    try:
        nested.rmdir()
    except OSError:
        pass


def download_gdrive_folder(
    url: str,
    target_dir: Path,
    *,
    force: bool,
    extract: bool,
    cleanup: bool,
) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        print("gdown not installed; run `uv sync`.", file=sys.stderr)
        return False

    before = {p.name for p in target_dir.iterdir()}
    if before and not force:
        print(f"Merging into existing folder: {target_dir}")
    else:
        print(f"Downloading Google Drive folder to {target_dir}...")
    gdown.download_folder(
        url=url,
        output=str(target_dir),
        quiet=False,
        remaining_ok=True,
        resume=not force,
    )
    _normalize_download_root(target_dir, before)

    if extract:
        extract_archives(target_dir, cleanup)
    return True


def download_hf_dataset(repo_id: str, target_dir: Path, *, force: bool) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed; run `uv sync`.", file=sys.stderr)
        return False

    if next(target_dir.iterdir(), None) is not None and not force:
        print(f"Merging into existing folder: {target_dir}")
    else:
        print(f"Downloading Hugging Face dataset {repo_id} to {target_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        force_download=force,
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download A2 assets, data, testing cases, and pretrained models."
    )
    parser.add_argument("--assets", action="store_true", help="Download assets to ./assets")
    parser.add_argument("--data", action="store_true", help="Download dataset to ./data")
    parser.add_argument(
        "--testing-cases",
        action="store_true",
        dest="testing_cases",
        help="Download testing cases to ./testing_cases",
    )
    parser.add_argument(
        "--pretrained-models",
        action="store_true",
        dest="pretrained_models",
        help="Download pretrained models to ./logs",
    )
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--force", action="store_true", help="Re-download even if target has content")
    parser.add_argument("--no-extract", action="store_true", help="Skip archive extraction")
    parser.add_argument("--cleanup", action="store_true", help="Remove archives after extraction")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all or not (args.assets or args.data or args.testing_cases or args.pretrained_models):
        args.assets = True
        args.data = True
        args.testing_cases = True
        args.pretrained_models = True

    ok = True
    if args.assets:
        ok &= download_gdrive_folder(
            ASSETS_URL,
            ROOT / "assets",
            force=args.force,
            extract=not args.no_extract,
            cleanup=args.cleanup,
        )
    if args.data:
        ok &= download_hf_dataset(HF_DATASET_REPO, ROOT / "data", force=args.force)
    if args.testing_cases:
        ok &= download_gdrive_folder(
            TESTING_CASES_URL,
            ROOT / "testing_cases",
            force=args.force,
            extract=not args.no_extract,
            cleanup=args.cleanup,
        )
    if args.pretrained_models:
        ok &= download_gdrive_folder(
            PRETRAINED_URL,
            ROOT / "logs",
            force=args.force,
            extract=not args.no_extract,
            cleanup=args.cleanup,
        )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
