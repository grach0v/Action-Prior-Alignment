"""
    Author: chenxi-wang
    Modified by Minghao Gou
"""

"""
GraspNet baseline wrapper that bridges the project to the upgraded
`GraspNet-PointNet2-Pytorch-General-Upgrade` codebase.

The upgraded repo lives in `models/GraspNet-PointNet2-Pytorch-General-Upgrade`
and supports modern Python/Torch/CUDA.  We add its submodules to `sys.path`
and reuse its network / collision utilities, while keeping the public API of
`GraspNetBaseLine` intact for the rest of the project.
"""

import os
import sys
from pathlib import Path
import numpy as np
import open3d as o3d

# compatibility shim for older deps expecting np.float
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import torch
from graspnetAPI import GraspGroup

# ---------------------------------------------------------------------------
# Wire up the upgraded GraspNet implementation.
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
UPGRADED_ROOT = THIS_DIR.parent / "GraspNet-PointNet2-Pytorch-General-Upgrade"
sys.path.extend(
    [
        str(UPGRADED_ROOT),
        str(UPGRADED_ROOT / "models"),
        str(UPGRADED_ROOT / "dataset"),
        str(UPGRADED_ROOT / "utils"),
    ]
)

from graspnet import GraspNet, pred_decode  # type: ignore  # noqa: E402
from graspnet_dataset import GraspNetDataset  # type: ignore  # noqa: E402
from collision_detector import ModelFreeCollisionDetector  # type: ignore  # noqa: E402
from data_utils import CameraInfo, create_point_cloud_from_depth_image  # type: ignore  # noqa: E402

class GraspNetBaseLine():
    def __init__(self, checkpoint_path, num_point = 20000, num_view = 300, collision_thresh = 0.001, empty_thresh = 0.15, voxel_size = 0.01):
        self.checkpoint_path = checkpoint_path
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.empty_thresh = empty_thresh
        self.voxel_size = voxel_size
        self.net = self.get_net()

    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net

    def get_and_process_data(self, cloud):
        cloud = cloud.voxel_down_sample(0.001)

        cloud_masked = np.asarray(cloud.points)
        color_masked = np.asarray(cloud.colors)

        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)

        objectness_score = end_points['objectness_score']
        from matplotlib import pyplot as plt
        
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def inference(self, o3d_pcd):
        end_points, cloud = self.get_and_process_data(o3d_pcd)
        gg = self.get_grasps(end_points)
        
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh, empty_thresh=self.empty_thresh)
        gg = gg[~collision_mask]
        return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])
