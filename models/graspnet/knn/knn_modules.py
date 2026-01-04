import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function

# Try CUDA knn, fall back to pure PyTorch if not available
try:
    from knn_pytorch import knn_pytorch
    _USE_CUDA_KNN = True
except ImportError:
    from models.knn_fallback import knn_pytorch
    _USE_CUDA_KNN = False
    import sys
    print("Note: Using pure PyTorch KNN fallback (no CUDA)", file=sys.stderr)


def knn(ref, query, k=1):
    """Compute k nearest neighbors for each query point."""
    device = ref.device
    ref = ref.float().to(device)
    query = query.float().to(device)
    inds = torch.empty(query.shape[0], k, query.shape[2]).long().to(device)
    knn_pytorch.knn(ref, query, inds)
    return inds
