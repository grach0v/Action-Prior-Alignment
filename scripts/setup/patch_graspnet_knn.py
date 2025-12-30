#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def patch_vision_h(path: Path) -> bool:
    text = _read(path)
    original = text
    if "THC/THC.h" in text:
        text = text.replace("#include <THC/THC.h>", "#include <cuda_runtime_api.h>")
    if "#include <cuda_runtime_api.h>" not in text:
        text = text.replace(
            "#include <torch/extension.h>",
            "#include <torch/extension.h>\n#include <cuda_runtime_api.h>",
        )
    if text != original:
        _write(path, text)
        return True
    return False


def patch_knn_h(path: Path) -> bool:
    text = _read(path)
    original = text

    if "THC/THC.h" in text:
        text = text.replace("#include <THC/THC.h>", "#include <c10/cuda/CUDAStream.h>")
    text = re.sub(r"^\s*extern THCState \*state;\n", "", text, flags=re.M)

    text = re.sub(r"\.data<([^>]+)>\(\)", r".data_ptr<\1>()", text)
    text = text.replace("ref.type().is_cuda()", "ref.is_cuda()")

    alloc_pattern = r"(\s*)float \*dist_dev = \(float\*\)THCudaMalloc\(state, ref_nb \* query_nb \* sizeof\(float\)\);\n"

    def _alloc_repl(match: re.Match[str]) -> str:
        indent = match.group(1)
        return (
            f"{indent}auto dist = at::empty({{ref_nb * query_nb}}, ref.options().dtype(at::kFloat));\n"
            f"{indent}float *dist_dev = dist.data_ptr<float>();\n"
            f"{indent}auto stream = c10::cuda::getCurrentCUDAStream(ref.get_device());\n"
        )

    text = re.sub(alloc_pattern, _alloc_repl, text)
    text = text.replace("THCudaFree(state, dist_dev);\n", "")
    text = text.replace("THCState_getCurrentStream(state)", "stream.stream()")
    text = text.replace("c10::cuda::getCurrentCUDAStream()", "stream.stream()")

    err_block = (
        r"\n\s*if \(err != cudaSuccess\)\s*\{\s*\n\s*printf\("
        r"\"error in knn: %s\\n\", cudaGetErrorString\(err\)\);\s*\n"
        r"\s*THError\(\"aborting\"\);\s*\n\s*\}"
    )
    text = re.sub(
        err_block,
        "\n    TORCH_CHECK(err == cudaSuccess, \"error in knn: \", cudaGetErrorString(err));",
        text,
        flags=re.S,
    )

    if text != original:
        _write(path, text)
        return True
    return False


def patch_setup_py(path: Path) -> bool:
    text = _read(path)
    original = text
    text = re.sub(
        r'^\s*"-gencode=arch=compute_90,code=sm_90",\n', "", text, flags=re.M
    )
    if text != original:
        _write(path, text)
        return True
    return False


def patch_graspnet_dataset(path: Path) -> bool:
    text = _read(path)
    original = text

    if "from torch._six import container_abcs" in text:
        text = text.replace(
            "from torch._six import container_abcs",
            "from collections.abc import Mapping, Sequence",
        )
    text = text.replace("container_abcs.Mapping", "Mapping")
    text = text.replace("container_abcs.Sequence", "Sequence")

    if text != original:
        _write(path, text)
        return True
    return False


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    knn_dir = root / "models" / "graspnet_new" / "knn"
    if not knn_dir.is_dir():
        print(f"error: {knn_dir} not found (did you clone GraspNet?)", file=sys.stderr)
        return 1

    changes = []
    vision_h = knn_dir / "src" / "cuda" / "vision.h"
    knn_h = knn_dir / "src" / "knn.h"
    setup_py = knn_dir / "setup.py"
    dataset_py = root / "models" / "graspnet_new" / "dataset" / "graspnet_dataset.py"

    if vision_h.exists():
        if patch_vision_h(vision_h):
            changes.append(str(vision_h))
    else:
        print(f"warn: missing {vision_h}", file=sys.stderr)

    if knn_h.exists():
        if patch_knn_h(knn_h):
            changes.append(str(knn_h))
    else:
        print(f"warn: missing {knn_h}", file=sys.stderr)

    if setup_py.exists():
        if patch_setup_py(setup_py):
            changes.append(str(setup_py))
    else:
        print(f"warn: missing {setup_py}", file=sys.stderr)

    if dataset_py.exists():
        if patch_graspnet_dataset(dataset_py):
            changes.append(str(dataset_py))
    else:
        print(f"warn: missing {dataset_py}", file=sys.stderr)

    if changes:
        print("patched:")
        for path in changes:
            print(f"  {path}")
    else:
        print("no changes needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
