import os
import tempfile
from pathlib import Path
from functools import lru_cache

import torch
from torch.utils.cpp_extension import CUDA_HOME, load, get_default_build_root


@lru_cache(maxsize=1)
def _load_knn_backend():
  base_dir = Path(__file__).resolve().parent
  src_dir = base_dir / "src"

  main_sources = list(src_dir.glob("*.cpp"))
  cpu_sources = list((src_dir / "cpu").glob("*.cpp"))
  cuda_sources = list((src_dir / "cuda").glob("*.cu"))

  sources = [str(p) for p in main_sources + cpu_sources]
  extra_cflags = ["-O3", "-std=c++17"]
  extra_cuda_cflags = None
  with_cuda = torch.cuda.is_available() and CUDA_HOME is not None and len(cuda_sources) > 0

  if with_cuda:
    sources += [str(p) for p in cuda_sources]
    extra_cflags.append("-DWITH_CUDA")
    extra_cuda_cflags = [
        "-O3",
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--std=c++17",
    ]

  build_root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", get_default_build_root()))
  try:
    build_root.mkdir(parents=True, exist_ok=True)
  except PermissionError:
    fallback = Path(tempfile.gettempdir()) / "a2_torch_extensions"
    fallback.mkdir(parents=True, exist_ok=True)
    build_root = fallback

  try:
    return load(
        name="a2_knn_pytorch",
        sources=sources,
        extra_include_paths=[str(src_dir)],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=with_cuda,
        build_directory=str(build_root),
        verbose=False,
    )
  except (RuntimeError, OSError) as exc:  # pragma: no cover - build failure guard
    raise RuntimeError(
        "Failed to build the knn CUDA extension. "
        "Verify that a compatible C++ toolchain is available and that the CUDA toolkit "
        "matches your PyTorch installation."
    ) from exc


def knn(ref, query, k=1):
  """ Compute k nearest neighbors for each query point.
  """
  backend = _load_knn_backend()
  device = ref.device
  ref = ref.float().to(device)
  query = query.float().to(device)
  inds = torch.empty(query.shape[0], k, query.shape[2], device=device, dtype=torch.long)
  backend.knn(ref, query, inds)
  return inds
