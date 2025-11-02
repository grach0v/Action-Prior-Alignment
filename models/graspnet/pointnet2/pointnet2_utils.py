# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import tempfile
import warnings
import torch
from torch.autograd import Function
import torch.nn as nn
from . import pytorch_utils as pt_utils
import sys
from pathlib import Path
from torch.utils.cpp_extension import CUDA_HOME, load, get_default_build_root

try:
    import builtins
except:  # pragma: no cover
    import __builtin__ as builtins

USE_POINTNET2_EXT = False
_POINTNET2_EXT_ERROR = None

try:
    import pointnet2._ext as _ext  # pragma: no cover
    USE_POINTNET2_EXT = True
except ImportError:
    if getattr(builtins, "__POINTNET2_SETUP__", False):
        raise

    _EXT_SRC_ROOT = Path(__file__).resolve().parent / "_ext_src"
    include_dir = _EXT_SRC_ROOT / "include"
    src_dir = _EXT_SRC_ROOT / "src"

    cpp_sources = [str(path) for path in src_dir.glob("*.cpp")]
    cuda_sources = [str(path) for path in src_dir.glob("*.cu")]

    extra_cflags = ["-O2", "-std=c++17", f"-I{include_dir}"]
    extra_cuda_cflags = ["-O2", "--std=c++17", f"-I{include_dir}"]

    build_with_cuda = (
        torch.cuda.is_available()
        and CUDA_HOME is not None
        and len(cuda_sources) > 0
    )

    sources = list(cpp_sources)
    if build_with_cuda:
        sources += cuda_sources
    else:
        extra_cuda_cflags = None

    build_root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", get_default_build_root()))
    try:
        build_root.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback = Path(tempfile.gettempdir()) / "a2_torch_extensions"
        fallback.mkdir(parents=True, exist_ok=True)
        build_root = fallback

    if not build_with_cuda:
        _POINTNET2_EXT_ERROR = RuntimeError(
            "CUDA is not available so the pointnet2 extension will not be built."
        )
        USE_POINTNET2_EXT = False
        warnings.warn(
            "CUDA runtime is unavailable; pointnet2 extension will not be built. "
            "Falling back to the pure PyTorch implementation."
        )
        _ext = None
    else:
        try:
            _ext = load(
                name="a2_pointnet2_ext",
                sources=sources,
                extra_include_paths=[str(include_dir)],
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                with_cuda=build_with_cuda,
                build_directory=str(build_root),
                verbose=False,
            )
            USE_POINTNET2_EXT = True
        except (RuntimeError, OSError) as exc:  # pragma: no cover - build failure guard
            _POINTNET2_EXT_ERROR = exc
            USE_POINTNET2_EXT = False
            warnings.warn(
                "PointNet++ CUDA extension could not be built; falling back to a pure "
                "PyTorch (CPU) implementation. Expect significantly slower execution.\n"
                f"Original error: {exc}"
            )

if not USE_POINTNET2_EXT:
    warnings.warn(
        "PointNet++ custom ops are running in CPU fallback mode. "
        "If you were expecting GPU acceleration, ensure that CUDA is available and "
        "reinstall Action-Prior-Alignment after fixing your toolchain."
    )


if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


def _fps_cpu(xyz, npoint):
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=torch.float32)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)  # (B,1,3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids


def _gather_points(features, idx):
    idx_expanded = idx.unsqueeze(1).expand(-1, features.size(1), -1)
    return torch.gather(features, 2, idx_expanded)


def _gather_points_grad(grad_out, idx, N=None):
    if N is None:
        N = int(idx.max().item()) + 1
    grad_points = torch.zeros(
        grad_out.size(0), grad_out.size(1), N,
        device=grad_out.device, dtype=grad_out.dtype
    )
    idx_expanded = idx.unsqueeze(1).expand(-1, grad_out.size(1), -1)
    grad_points.scatter_add_(2, idx_expanded, grad_out)
    return grad_points


def _group_points(features, idx):
    idx_expanded = idx.unsqueeze(1).expand(-1, features.size(1), -1, -1)
    features_expand = features.unsqueeze(2).expand(-1, -1, idx.size(1), -1)
    return torch.gather(features_expand, 3, idx_expanded)


def _group_points_grad(grad_out, idx, N):
    grad_points = torch.zeros(
        grad_out.size(0), grad_out.size(1), N,
        device=grad_out.device, dtype=grad_out.dtype
    )
    idx_expanded = idx.unsqueeze(1).expand(-1, grad_out.size(1), -1, -1)
    grad_points.scatter_add_(
        2,
        idx_expanded.reshape(idx.size(0), grad_out.size(1), -1),
        grad_out.reshape(idx.size(0), grad_out.size(1), -1),
    )
    return grad_points


def _ball_query_cpu(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device
    idx = torch.zeros(B, S, nsample, dtype=torch.long, device=device)
    dist = torch.cdist(new_xyz, xyz, p=2)
    for b in range(B):
        for s in range(S):
            mask = dist[b, s] <= radius
            inds = torch.nonzero(mask, as_tuple=False).flatten()
            if inds.numel() == 0:
                inds = torch.argsort(dist[b, s])[:1]
            if inds.numel() >= nsample:
                idx[b, s] = inds[:nsample]
            else:
                idx[b, s, :inds.numel()] = inds
                idx[b, s, inds.numel():] = inds[-1]
    return idx


def _three_nn_cpu(unknown, known):
    dist = torch.cdist(unknown, known, p=2)
    dists, idx = torch.topk(dist, k=3, dim=2, largest=False, sorted=True)
    return dists, idx


def _three_interpolate_cpu(features, idx, weight):
    B, C, M = features.shape
    _, N, _ = idx.shape
    idx_expand = idx.permute(0, 2, 1).contiguous().view(B, 3, N)
    gathered = torch.gather(
        features.unsqueeze(2).expand(-1, -1, N, -1),
        3,
        idx.unsqueeze(1).expand(-1, C, -1, -1)
    )
    weighted = gathered * weight.unsqueeze(1)
    return torch.sum(weighted, dim=-1)


def _three_interpolate_grad_cpu(grad_out, idx, weight, m):
    B, C, N = grad_out.shape
    grad_features = torch.zeros(B, C, m, device=grad_out.device, dtype=grad_out.dtype)
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)
    grad_features.scatter_add_(2, idx_expanded, grad_out.unsqueeze(-1) * weight.unsqueeze(1))
    return grad_features


def _cylinder_query_cpu(radius, hmin, hmax, nsample, xyz, new_xyz, rot):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device
    rot = rot.view(B, S, 3, 3)
    idx = torch.zeros(B, S, nsample, dtype=torch.long, device=device)

    for b in range(B):
        for s in range(S):
            center = new_xyz[b, s]
            R = rot[b, s]
            diff = xyz[b] - center
            local = diff @ R  # convert to cylinder frame
            radial = torch.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
            mask = (radial <= radius) & (local[:, 2] >= hmin) & (local[:, 2] <= hmax)
            inds = torch.nonzero(mask, as_tuple=False).flatten()
            if inds.numel() == 0:
                inds = torch.argsort(radial)[:1]
            if inds.numel() >= nsample:
                idx[b, s] = inds[:nsample]
            else:
                idx[b, s, :inds.numel()] = inds
                idx[b, s, inds.numel():] = inds[-1]
    return idx


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        if USE_POINTNET2_EXT:
            return _ext.furthest_point_sampling(xyz, npoint)
        return _fps_cpu(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        if USE_POINTNET2_EXT:
            ctx.for_backwards = (idx, C, N)
            return _ext.gather_points(features, idx)

        ctx.save_for_backward(idx)
        return _gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        if USE_POINTNET2_EXT:
            idx, C, N = ctx.for_backwards
            grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
            return grad_features, None

        (idx,) = ctx.saved_tensors
        grad_features = _gather_points_grad(grad_out, idx, grad_out.size(-1))
        return grad_features, None


if USE_POINTNET2_EXT:
    gather_operation = GatherOperation.apply
else:
    def gather_operation(features, idx):
        return _gather_points(features, idx)


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        if USE_POINTNET2_EXT:
            dist2, idx = _ext.three_nn(unknown, known)
        else:
            dist2, idx = _three_nn_cpu(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


if USE_POINTNET2_EXT:
    three_nn = ThreeNN.apply
else:
    def three_nn(unknown, known):
        dist, idx = _three_nn_cpu(unknown, known)
        return torch.sqrt(dist), idx


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        if USE_POINTNET2_EXT:
            return _ext.three_interpolate(features, idx, weight)

        return _three_interpolate_cpu(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        if USE_POINTNET2_EXT:
            grad_features = _ext.three_interpolate_grad(
                grad_out.contiguous(), idx, weight, m
            )
        else:
            grad_features = _three_interpolate_grad_cpu(
                grad_out.contiguous(), idx, weight, m
            )

        return grad_features, None, None


if USE_POINTNET2_EXT:
    three_interpolate = ThreeInterpolate.apply
else:
    def three_interpolate(features, idx, weight):
        return _three_interpolate_cpu(features, idx, weight)


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        if USE_POINTNET2_EXT:
            ctx.for_backwards = (idx, N)
            return _ext.group_points(features, idx)

        ctx.save_for_backward(idx)
        return _group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        if USE_POINTNET2_EXT:
            idx, N = ctx.for_backwards
            grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
            return grad_features, None

        (idx,) = ctx.saved_tensors
        N = int(idx.max().item()) + 1
        grad_features = _group_points_grad(grad_out, idx, N)
        return grad_features, None


if USE_POINTNET2_EXT:
    grouping_operation = GroupingOperation.apply
else:
    def grouping_operation(features, idx):
        return _group_points(features, idx)


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        if USE_POINTNET2_EXT:
            return _ext.ball_query(new_xyz, xyz, radius, nsample)
        return _ball_query_cpu(radius, nsample, xyz, new_xyz)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


if USE_POINTNET2_EXT:
    ball_query = BallQuery.apply
else:
    def ball_query(radius, nsample, xyz, new_xyz):
        return _ball_query_cpu(radius, nsample, xyz, new_xyz)


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]), device=idx.device)
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long, device=idx.device)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features


class CylinderQuery(Function):
    @staticmethod
    def forward(ctx, radius, hmin, hmax, nsample, xyz, new_xyz, rot):
        # type: (Any, float, float, float, int, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the cylinders
        hmin, hmax : float
            endpoints of cylinder height in x-rotation axis
        nsample : int
            maximum number of features in the cylinders
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the cylinder query
        rot: torch.Tensor
            (B, npoint, 9) flatten rotation matrices from
                           cylinder frame to world frame

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        if USE_POINTNET2_EXT:
            return _ext.cylinder_query(new_xyz, xyz, rot, radius, hmin, hmax, nsample)
        return _cylinder_query_cpu(radius, hmin, hmax, nsample, xyz, new_xyz, rot)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None


if USE_POINTNET2_EXT:
    cylinder_query = CylinderQuery.apply
else:
    def cylinder_query(radius, hmin, hmax, nsample, xyz, new_xyz, rot):
        return _cylinder_query_cpu(radius, hmin, hmax, nsample, xyz, new_xyz, rot)


class CylinderQueryAndGroup(nn.Module):
    r"""
    Groups with a cylinder query of radius and height

    Parameters
    ---------
    radius : float32
        Radius of cylinder
    hmin, hmax: float32
        endpoints of cylinder height in x-rotation axis
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, hmin, hmax, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, rotate_xyz=True, sample_uniformly=False, ret_unique_cnt=False):
        # type: (CylinderQueryAndGroup, float, float, float, int, bool) -> None
        super(CylinderQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.hmin, self.hmax, = radius, nsample, hmin, hmax
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.rotate_xyz = rotate_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, rot, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        rot : torch.Tensor
            rotation matrices (B, npoint, 3, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        B, npoint, _ = new_xyz.size()
        idx = cylinder_query(self.radius, self.hmin, self.hmax, self.nsample, xyz, new_xyz, rot.view(B, npoint, 9))

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]), device=idx.device)
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long, device=idx.device)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if self.rotate_xyz:
            grouped_xyz_ = grouped_xyz.permute(0, 2, 3, 1).contiguous() # (B, npoint, nsample, 3)
            grouped_xyz_ = torch.matmul(grouped_xyz_, rot)
            grouped_xyz = grouped_xyz_.permute(0, 3, 1, 2).contiguous()


        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)
