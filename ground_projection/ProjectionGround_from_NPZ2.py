#!/usr/bin/env python
# coding: utf-8
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ground_projection.util.semantic_grid import SemanticGrid
from ground_projection.util import viz_utils, map_utils, utils, load_slam_dataset
from ground_projection.publish3D_util import publish3D_rviz

if __name__ == '__main__':
    # 加载数据
    npz_file_path = "/home/robotlab/work/semantic-segmentation-pytorch/save_results/all_data.npz"
    data = np.load(npz_file_path)
    all_sseg = data["ssegs"]       # (N, H, W)
    all_depth = data["depth_imgs"] # (N, H, W)
    all_pose = data["abs_pose"]    # (N, 3)

    # 地图参数
    spatial_labels = 3
    object_labels = 27
    grid_dim = (400, 400)
    cell_size = 0.1
    crop_size = (64, 64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化全局地图张量
    global_grid = torch.zeros((1, spatial_labels, *grid_dim), dtype=torch.float32).to("cuda")
    # For each episode we need a new instance of a fresh global grid
    sg = SemanticGrid(1, grid_dim, crop_size[0], cell_size,
                      spatial_labels=spatial_labels, object_labels=object_labels)
    abs_poses = []

    for i in range(len(all_sseg)):
        depth = np.squeeze(all_depth[i])
        sseg = np.squeeze(all_sseg[i])
        camera_position = np.squeeze(all_pose[i])

        # 相机 → 机器人坐标系变换
        # 右手坐标系
        First_Camera_Pose = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        # 左手坐标系（ICL）
        First_Camera_Pose_ICL = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])

        First_Camera_t = np.array([0, 0, 1.17])

        # mask = (sseg != 17)
        # if not np.any(mask):
        #     continue
        mask = (sseg > -1)

        print("帧 {}，位姿: {}".format(i, camera_position))

        # 相机点云和像素坐标
        points_cam, u, v = publish3D_rviz.depth_to_pointcloud(depth, mask)
        points_in_FirstFrameCoordinate= publish3D_rviz.transform_pointcloud_to_firstFrameCoordinate(points_cam, camera_position)
        points_world = points_in_FirstFrameCoordinate@First_Camera_Pose_ICL + First_Camera_t
        local3D = points_world[np.newaxis, ...]  # shape: [1, N, 3]
        local3D = torch.from_numpy(local3D).float().to("cuda")

        points2D = np.stack((u, v), axis=-1)[np.newaxis, ...]  # shape: [1, N, 2]
        points2D = torch.from_numpy(points2D).float().to("cuda")

        ssegs_3 = sseg[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        ssegs_3 = torch.from_numpy(ssegs_3).float().to("cuda")

        # 构建单帧语义栅格
        ego_semantic_sseg_27 = map_utils.ground_projection_my(
            points2D, local3D, ssegs_3,
            sseg_labels=object_labels,
            grid_dim=grid_dim,
            cell_size=cell_size
        )  # shape: [t, 27, 184, 184]
        print("ego_grid_sseg_3.shape: ", ego_semantic_sseg_27.shape)

        # FIXME: SLAM中的ego_semantic_sseg_27都是world坐标系下的。因此下文的全局更新时，设置_rel_pose和abs_poses为0即可。

        # 累加到全局地图
        position_zero = [0,0,0]
        abs_poses.append(position_zero)
        # rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
        _rel_pose = torch.Tensor(position_zero).unsqueeze(0).float()
        _rel_pose = _rel_pose.to(device)
        geo_semantic_sseg = sg.mapTransformer(grid=ego_semantic_sseg_27, pose=_rel_pose,
                                              abs_pose=torch.tensor(abs_poses).to(device))
        step_geo_grid_sseg = sg.update_semantic_proj_grid_bayes(geo_grid=geo_semantic_sseg.unsqueeze(0))

        # 从世界坐标系转移到机器人坐标系
        step_ego_grid_sseg = sg.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=_rel_pose,
                                           abs_pose=torch.tensor(abs_poses).to(device))


        # 可视化
        flag = 2
        if(flag == 1):
            ego_semantic_sseg_27 = ego_semantic_sseg_27.squeeze(0)
            label_map = ego_semantic_sseg_27.argmax(dim=0).cpu().numpy()  # shape: (H, W)
        if (flag == 2):
            step_geo_grid_sseg = step_geo_grid_sseg.squeeze(0).squeeze(0)
            label_map = step_geo_grid_sseg.argmax(dim=0).cpu().numpy()  # shape: (H, W)

        ego_vis = viz_utils.colorEncode(label_map)

        # 可视化融合地图
        plt.figure(figsize=(10, 10))
        plt.imshow(ego_vis)
        plt.title("Fused Semantic Map (10 Frames)")
        plt.axis("off")
        plt.tight_layout()
        # plt.show()
        # 保存到本地的/home/robotlab/work/semantic-segmentation-pytorch/save_results/temp_projection文件夹下
        # plt.savefig("保存路径/文件名.png")
        output_dir = "/home/robotlab/work/semantic-segmentation-pytorch/save_results/temp_projection"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"frame_{i:02d}.png")
        plt.savefig(save_path, dpi=300)
