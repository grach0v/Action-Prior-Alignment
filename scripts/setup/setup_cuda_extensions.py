#!/usr/bin/env python3
"""
Post-sync setup script for GraspNet and CUDA extensions.

This script:
1. Clones the GraspNet baseline if not present
2. Patches files for modern PyTorch compatibility
3. Builds the pointnet2 and knn CUDA extensions (Linux/CUDA only)

Run after `uv sync`:
    uv run python scripts/setup/setup_cuda_extensions.py

On macOS, CUDA extensions are skipped and pure PyTorch fallbacks are used instead.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRASPNET_DIR = ROOT / "models" / "graspnet_new"
GRASPNET_REPO = "https://github.com/H-Freax/GraspNet-PointNet2-Pytorch-General-Upgrade.git"

IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and print it."""
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check)


def clone_graspnet() -> bool:
    """Clone the GraspNet baseline if not present."""
    if GRASPNET_DIR.is_dir() and (GRASPNET_DIR / "pointnet2").is_dir():
        print(f"✓ GraspNet already cloned at {GRASPNET_DIR}")
        return True

    print(f"Cloning GraspNet baseline to {GRASPNET_DIR}...")
    GRASPNET_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Remove if exists but incomplete
    if GRASPNET_DIR.exists():
        import shutil
        shutil.rmtree(GRASPNET_DIR)

    run(["git", "clone", GRASPNET_REPO, str(GRASPNET_DIR)])
    return True


def run_patches() -> bool:
    """Run the patch script for modern PyTorch compatibility."""
    patch_script = ROOT / "scripts" / "setup" / "patch_graspnet_knn.py"
    if not patch_script.exists():
        print(f"✗ Patch script not found: {patch_script}")
        return False

    print("Running patches for modern PyTorch compatibility...")
    run([sys.executable, str(patch_script)])
    return True


def build_pointnet2() -> bool:
    """Build the pointnet2 CUDA extension (Linux only)."""
    if IS_MACOS:
        print("⊘ Skipping pointnet2 CUDA build on macOS (using pure PyTorch fallback)")
        return True

    pointnet2_dir = GRASPNET_DIR / "pointnet2"
    if not pointnet2_dir.is_dir():
        print(f"✗ pointnet2 directory not found: {pointnet2_dir}")
        return False

    print(f"\nBuilding pointnet2 CUDA extension...")

    # Check if already built
    try:
        import pointnet2
        print("✓ pointnet2 already installed")
        return True
    except ImportError:
        pass

    # Build with --no-build-isolation since setup.py imports torch
    # Use uv pip install since uv-managed environments don't have pip
    run(
        ["uv", "pip", "install", ".", "--no-build-isolation", "-v"],
        cwd=pointnet2_dir
    )
    return True


def build_knn() -> bool:
    """Build the knn CUDA extension (Linux only)."""
    if IS_MACOS:
        print("⊘ Skipping knn CUDA build on macOS (using pure PyTorch fallback)")
        return True

    knn_dir = GRASPNET_DIR / "knn"
    if not knn_dir.is_dir():
        print(f"✗ knn directory not found: {knn_dir}")
        return False

    print(f"\nBuilding knn CUDA extension...")

    # Check if already built
    try:
        import knn_pytorch
        print("✓ knn_pytorch already installed")
        return True
    except ImportError:
        pass

    # Build with --no-build-isolation since setup.py imports torch
    # Use uv pip install since uv-managed environments don't have pip
    run(
        ["uv", "pip", "install", ".", "--no-build-isolation", "-v"],
        cwd=knn_dir
    )
    return True


def download_checkpoint() -> bool:
    """Download the GraspNet checkpoint if not present."""
    checkpoint_path = GRASPNET_DIR / "checkpoint-rs.tar"
    if checkpoint_path.exists():
        print(f"✓ Checkpoint already exists at {checkpoint_path}")
        return True

    print("\nDownloading GraspNet checkpoint...")
    try:
        import gdown
        gdown.download(
            "https://drive.google.com/uc?id=1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk",
            str(checkpoint_path),
            quiet=False
        )
        return True
    except Exception as e:
        print(f"⚠ Could not download checkpoint: {e}")
        print("  You can download manually from:")
        print("  https://drive.google.com/uc?id=1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk")
        return False


def verify_installation() -> bool:
    """Verify that all components are installed correctly."""
    print("\n" + "=" * 60)
    print("Verifying installation...")
    print("=" * 60)

    errors = []

    # Check torch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
        else:
            print(f"⚠ PyTorch {torch.__version__} (CUDA not available)")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")

    # Check pointnet2
    try:
        import pointnet2
        print("✓ pointnet2 installed")
    except ImportError as e:
        errors.append(f"pointnet2: {e}")

    # Check knn
    try:
        import knn_pytorch
        print("✓ knn_pytorch installed")
    except ImportError as e:
        errors.append(f"knn_pytorch: {e}")

    # Check a2 package
    try:
        import a2
        print("✓ a2 package installed")
    except ImportError as e:
        errors.append(f"a2: {e}")

    # Check GraspNet
    graspnet_baseline = GRASPNET_DIR / "graspnet_baseline.py"
    graspnet_model = GRASPNET_DIR / "models" / "graspnet.py"
    if graspnet_baseline.exists() or graspnet_model.exists():
        print("✓ GraspNet baseline present")
    else:
        errors.append(
            "GraspNet baseline not found (expected graspnet_baseline.py or models/graspnet.py)"
        )

    if errors:
        print("\n⚠ Some components failed:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("\n✓ All components installed successfully!")
    return True


def main() -> int:
    print("=" * 60)
    print("A2 - GraspNet Setup")
    print("=" * 60)
    print(f"Platform: {platform.system()} ({platform.machine()})")

    if IS_MACOS:
        print("\n📱 macOS detected - will use pure PyTorch fallbacks for PointNet2 ops")
        print("   (CUDA extensions will be skipped)")
    else:
        # Check CUDA is available
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if not cuda_home:
            # Try common paths
            for path in ["/usr/local/cuda", "/usr/local/cuda-12.4", "/usr/local/cuda-12.1"]:
                if Path(path).exists():
                    cuda_home = path
                    break

        if cuda_home:
            print(f"CUDA_HOME: {cuda_home}")
            os.environ["CUDA_HOME"] = cuda_home
        else:
            print("⚠ CUDA_HOME not set. CUDA extensions may fail to build.")
            print("  Set CUDA_HOME to your CUDA installation (e.g., /usr/local/cuda-12.4)")

    # Step 1: Clone GraspNet
    if not clone_graspnet():
        return 1

    # Step 2: Run patches
    if not run_patches():
        return 1

    # Step 3: Build CUDA extensions
    if not build_pointnet2():
        print("⚠ pointnet2 build failed, but continuing...")

    if not build_knn():
        print("⚠ knn build failed, but continuing...")

    # Step 4: Download checkpoint (optional)
    download_checkpoint()

    # Step 5: Verify installation
    verify_installation()

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
