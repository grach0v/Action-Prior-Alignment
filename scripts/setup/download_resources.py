#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]

ASSETS_URL = "https://drive.google.com/drive/folders/1WxKDFXJktoqiP0jmkDZrMCcNNBx5u-YM?usp=drive_link"
HF_ASSETS_REPO = "dgrachev/a2_assets"
TESTING_CASES_URL = "https://drive.google.com/drive/folders/1OuTua-69NEeV7RYIi9nzR1jmdZEugB68?usp=sharing"
PRETRAINED_URL = "https://drive.google.com/drive/folders/1uoDGIgkcSi8okcr8qjKOaF57TyRaHRd_?usp=sharing"
HF_DATASET_REPO = "KechunXu1/A2_Dataset"
GDRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"


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


def _parse_gdrive_folder_id(url: str) -> str | None:
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None


def _drive_api_list_children(folder_id: str, api_key: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    page_token: str | None = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id, name, mimeType)",
            "pageSize": "1000",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        url = "https://www.googleapis.com/drive/v3/files?" + urlencode(params)
        try:
            with urlopen(url) as resp:
                data = json.load(resp)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Drive API error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Drive API request failed: {exc}") from exc

        items.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return items


def _drive_api_walk(folder_id: str, api_key: str, prefix: Path):
    for item in _drive_api_list_children(folder_id, api_key):
        name = item.get("name", "unnamed").replace(os.sep, "_")
        item_id = item.get("id")
        if not item_id:
            continue
        if item.get("mimeType") == GDRIVE_FOLDER_MIME:
            yield from _drive_api_walk(item_id, api_key, prefix / name)
        else:
            yield item_id, prefix / name


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
    show_progress: bool = False,
    progress_label: str = "Download",
    api_key: str | None = None,
) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        print("gdown not installed; run `uv sync`.", file=sys.stderr)
        return False

    if api_key:
        folder_id = _parse_gdrive_folder_id(url)
        if not folder_id:
            print("Warning: could not parse Google Drive folder ID, using gdown fallback.")
        else:
            try:
                files = list(_drive_api_walk(folder_id, api_key, Path(".")))
            except RuntimeError as exc:
                print(f"Drive API listing failed, using gdown fallback: {exc}")
                files = None

            if files is not None:
                pending = []
                existing = 0
                for file_id, rel_path in files:
                    local_path = target_dir / rel_path
                    if not force and local_path.exists():
                        existing += 1
                        continue
                    pending.append((file_id, rel_path, local_path))

                if existing:
                    print(f"Already present: {existing} files in {target_dir}")
                if not pending:
                    print(f"No files to download for {progress_label.lower()}")
                else:
                    try:
                        from tqdm import tqdm
                    except ImportError:
                        tqdm = None

                    total = len(pending)
                    if show_progress and tqdm is not None:
                        with tqdm(total=total, desc=progress_label, unit="file", ascii=True) as pbar:
                            for file_id, rel_path, local_path in pending:
                                local_path.parent.mkdir(parents=True, exist_ok=True)
                                result = gdown.download(
                                    url=f"https://drive.google.com/uc?id={file_id}",
                                    output=str(local_path),
                                    quiet=True,
                                    resume=not force,
                                )
                                if result is None:
                                    print(f"Failed to download {rel_path}", file=sys.stderr)
                                    return False
                                pbar.update(1)
                    else:
                        print(f"Downloading {total} files for {progress_label.lower()}...")
                        for idx, (file_id, rel_path, local_path) in enumerate(pending, 1):
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            result = gdown.download(
                                url=f"https://drive.google.com/uc?id={file_id}",
                                output=str(local_path),
                                quiet=True,
                                resume=not force,
                            )
                            if result is None:
                                print(f"Failed to download {rel_path}", file=sys.stderr)
                                return False
                            if show_progress:
                                print(f"[{idx}/{total}] {rel_path}")

                if extract:
                    extract_archives(target_dir, cleanup)
                return True

    before = {p.name for p in target_dir.iterdir()}
    if show_progress:
        print(f"Scanning Google Drive folder for {progress_label.lower()}...")
        files = gdown.download_folder(
            url=url,
            output=str(target_dir),
            quiet=True,
            remaining_ok=True,
            skip_download=True,
        )
        if files is None:
            print("Failed to retrieve folder contents.", file=sys.stderr)
            return False

        print(
            "Note: gdown listings are limited to ~50 items per folder. "
            "If downloads stop early, set GDRIVE_API_KEY and rerun."
        )

        pending = []
        existing = 0
        for item in files:
            local_path = Path(item.local_path)
            if not force and local_path.exists():
                existing += 1
                continue
            pending.append(item)

        if existing:
            print(f"Already present: {existing} files in {target_dir}")
        if not pending:
            print(f"No files to download for {progress_label.lower()}")
        else:
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = None

            total = len(pending)
            if tqdm is None:
                print(f"Downloading {total} files for {progress_label.lower()}...")
                for idx, item in enumerate(pending, 1):
                    local_path = Path(item.local_path)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    result = gdown.download(
                        url=f"https://drive.google.com/uc?id={item.id}",
                        output=str(local_path),
                        quiet=True,
                        resume=not force,
                    )
                    if result is None:
                        print(f"Failed to download {item.path}", file=sys.stderr)
                        return False
                    print(f"[{idx}/{total}] {item.path}")
            else:
                with tqdm(total=total, desc=progress_label, unit="file", ascii=True) as pbar:
                    for item in pending:
                        local_path = Path(item.local_path)
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        result = gdown.download(
                            url=f"https://drive.google.com/uc?id={item.id}",
                            output=str(local_path),
                            quiet=True,
                            resume=not force,
                        )
                        if result is None:
                            print(f"Failed to download {item.path}", file=sys.stderr)
                            return False
                        pbar.update(1)
    else:
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


def download_hf_dataset(
    repo_id: str,
    target_dir: Path,
    *,
    force: bool,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    label: str = "dataset",
) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed; run `uv sync`.", file=sys.stderr)
        return False

    if next(target_dir.iterdir(), None) is not None and not force:
        print(f"Merging into existing folder: {target_dir}")
    else:
        print(f"Downloading Hugging Face {label} {repo_id} to {target_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        force_download=force,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download A2 assets, data, testing cases, and pretrained models."
    )
    parser.add_argument("--assets", action="store_true", help="Download assets to ./assets")
    parser.add_argument(
        "--assets-source",
        choices=["hf", "gdrive"],
        default="hf",
        help="Where to download assets from (default: hf).",
    )
    parser.add_argument(
        "--assets-repo",
        default=None,
        help="Hugging Face dataset repo for assets (default: dgrachev/a2_assets).",
    )
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
        help="Download pretrained models to ./logs/a2_pretrained",
    )
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--force", action="store_true", help="Re-download even if target has content")
    parser.add_argument("--no-extract", action="store_true", help="Skip archive extraction")
    parser.add_argument("--cleanup", action="store_true", help="Remove archives after extraction")
    parser.add_argument(
        "--gdrive-api-key",
        default=None,
        help="Google Drive API key (or set GDRIVE_API_KEY) to bypass the 50-item folder limit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all or not (args.assets or args.data or args.testing_cases or args.pretrained_models):
        args.assets = True
        args.data = True
        args.testing_cases = True
        args.pretrained_models = True

    gdrive_api_key = args.gdrive_api_key or os.environ.get("GDRIVE_API_KEY")
    assets_repo = args.assets_repo or os.environ.get("A2_ASSETS_REPO") or HF_ASSETS_REPO

    ok = True
    if args.assets:
        if args.assets_source == "gdrive":
            ok &= download_gdrive_folder(
                ASSETS_URL,
                ROOT / "assets",
                force=args.force,
                extract=not args.no_extract,
                cleanup=args.cleanup,
                show_progress=True,
                progress_label="Assets",
                api_key=gdrive_api_key,
            )
        else:
            ok &= download_hf_dataset(
                assets_repo,
                ROOT / "assets",
                force=args.force,
                allow_patterns=["simplified_objects/**", "unseen_objects/**"],
                label="assets dataset",
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
            api_key=gdrive_api_key,
        )
    if args.pretrained_models:
        ok &= download_gdrive_folder(
            PRETRAINED_URL,
            ROOT / "logs" / "a2_pretrained",
            force=args.force,
            extract=not args.no_extract,
            cleanup=args.cleanup,
            api_key=gdrive_api_key,
        )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
