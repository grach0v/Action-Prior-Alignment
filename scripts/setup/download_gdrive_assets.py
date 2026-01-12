#!/usr/bin/env python3
"""
Download assets from Google Drive and check for textures.
Uses gdown for Google Drive downloads.

Usage:
    python scripts/setup/download_gdrive_assets.py --check-only  # Just check current textures
    python scripts/setup/download_gdrive_assets.py --download    # Download and check
"""

import os
import re
import argparse
import subprocess
from pathlib import Path


def extract_gdrive_id(url: str) -> tuple[str, str]:
    """Extract Google Drive ID and type (folder/file) from URL."""
    # Folder: https://drive.google.com/drive/folders/{id}?...
    folder_match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    if folder_match:
        return folder_match.group(1), 'folder'

    # File: https://drive.google.com/file/d/{id}/...
    file_match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if file_match:
        return file_match.group(1), 'file'

    return None, None


def download_gdrive_folder(folder_id: str, output_dir: str) -> bool:
    """Download a Google Drive folder using gdown."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        subprocess.run(
            ["gdown", "--folder", url, "-O", output_dir, "--remaining-ok"],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error downloading folder {folder_id}: {e.stderr}")
        return False


def download_gdrive_file(file_id: str, output_path: str) -> bool:
    """Download a Google Drive file using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        subprocess.run(
            ["gdown", url, "-O", output_path],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error downloading file {file_id}: {e.stderr}")
        return False


def check_textures(assets_dir: str) -> dict:
    """Check which objects have textures."""
    assets_path = Path(assets_dir)
    results = {
        'with_texture': [],
        'without_texture': [],
        'missing_dir': [],
        'texture_files': {}  # Maps object name to texture file found
    }

    # Texture file candidates (in priority order)
    texture_candidates = [
        'texture_map.png',
        'textured.png',
        'textured.jpg',
        'texture.png',
        'texture.jpg',
    ]

    # Check for directories (object folders)
    for item in sorted(assets_path.iterdir()):
        if item.is_dir():
            found_texture = None
            for tex_name in texture_candidates:
                texture_file = item / tex_name
                if texture_file.exists():
                    found_texture = tex_name
                    break

            if found_texture:
                results['with_texture'].append(item.name)
                results['texture_files'][item.name] = found_texture
            else:
                results['without_texture'].append(item.name)

    return results


def read_gdrive_links(links_file: str) -> list[tuple[str, str]]:
    """Read Google Drive links from file."""
    links = []
    with open(links_file, 'r') as f:
        for line in f:
            url = line.strip()
            if url:
                gdrive_id, link_type = extract_gdrive_id(url)
                if gdrive_id:
                    links.append((gdrive_id, link_type, url))
    return links


def download_assets(links_file: str, output_dir: str, temp_dir: str = "/tmp/gdrive_assets"):
    """Download assets from Google Drive links file."""
    links = read_gdrive_links(links_file)

    # Separate folders and files
    folders = [(gid, url) for gid, lt, url in links if lt == 'folder']
    files = [(gid, url) for gid, lt, url in links if lt == 'file']

    print(f"Found {len(folders)} folders and {len(files)} files in {links_file}")

    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)

    # Download folders
    print(f"\nDownloading {len(folders)} folders...")
    for i, (folder_id, url) in enumerate(folders):
        print(f"  [{i+1}/{len(folders)}] Downloading folder {folder_id}...")
        folder_output = os.path.join(temp_dir, f"folder_{i:03d}")
        os.makedirs(folder_output, exist_ok=True)
        download_gdrive_folder(folder_id, folder_output)

    # Download files (URDFs)
    print(f"\nDownloading {len(files)} files...")
    for i, (file_id, url) in enumerate(files):
        print(f"  [{i+1}/{len(files)}] Downloading file {file_id}...")
        file_output = os.path.join(temp_dir, f"file_{i:03d}")
        download_gdrive_file(file_id, file_output)

    return temp_dir


def compare_textures(current_dir: str, downloaded_dir: str):
    """Compare textures between current and downloaded assets."""
    current = check_textures(current_dir)

    # Check downloaded folders for textures
    downloaded_path = Path(downloaded_dir)
    new_textures = []

    for folder in downloaded_path.glob("folder_*"):
        if folder.is_dir():
            for item in folder.iterdir():
                if item.is_dir():
                    texture_file = item / 'texture_map.png'
                    if texture_file.exists():
                        # Check if this object exists in current but without texture
                        obj_name = item.name
                        if obj_name in current['without_texture']:
                            new_textures.append((obj_name, str(texture_file)))
                            print(f"  Found new texture for {obj_name}: {texture_file}")

    return new_textures


def main():
    parser = argparse.ArgumentParser(description='Download and check assets from Google Drive')
    parser.add_argument('--check-only', action='store_true', help='Only check current textures')
    parser.add_argument('--download', action='store_true', help='Download assets from Google Drive')
    parser.add_argument('--simplified', default='gdrive_assets_simplified_objects.txt', help='Simplified objects links file')
    parser.add_argument('--unseen', default='gdrive_assets_unseen_objects.txt', help='Unseen objects links file')
    parser.add_argument('--assets-dir', default='assets/simplified_objects', help='Current assets directory')
    args = parser.parse_args()

    # Check current textures
    print("=" * 60)
    print("Checking current textures in", args.assets_dir)
    print("=" * 60)

    current = check_textures(args.assets_dir)
    print(f"\nObjects with texture ({len(current['with_texture'])}):")
    for obj in current['with_texture'][:20]:
        print(f"  {obj}")
    if len(current['with_texture']) > 20:
        print(f"  ... and {len(current['with_texture']) - 20} more")

    print(f"\nObjects without texture ({len(current['without_texture'])}):")
    for obj in current['without_texture'][:20]:
        print(f"  {obj}")
    if len(current['without_texture']) > 20:
        print(f"  ... and {len(current['without_texture']) - 20} more")

    if args.check_only:
        return

    if args.download:
        print("\n" + "=" * 60)
        print("Downloading assets from Google Drive")
        print("=" * 60)

        # Download simplified objects
        if os.path.exists(args.simplified):
            print(f"\nProcessing {args.simplified}...")
            temp_dir = download_assets(args.simplified, args.assets_dir)

            print("\nComparing textures...")
            new_textures = compare_textures(args.assets_dir, temp_dir)

            if new_textures:
                print(f"\nFound {len(new_textures)} new textures!")
                for obj_name, tex_path in new_textures:
                    print(f"  {obj_name}: {tex_path}")
            else:
                print("\nNo new textures found.")
        else:
            print(f"Links file not found: {args.simplified}")


if __name__ == '__main__':
    main()
