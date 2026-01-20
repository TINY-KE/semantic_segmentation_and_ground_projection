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
    path = "/home/robotlab/dataset/semantic/semantic_datasets/data_v6/test_old/2azQ1b91cZZ/ep_1_1_2azQ1b91cZZ.npz"
    data = np.load(path)
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

    for i in range(10):
        depth = np.squeeze(all_depth[i])
        sseg = np.squeeze(all_sseg[i])
        print("sseg.shape: ", sseg.shape)
        position = np.squeeze(all_pose[i])
        position = -1 * position
        print("position: ", position)
        print("position.shape: ", position.shape)

        # 相机 → 机器人坐标变换
        R_robot_to_cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        # mask = (sseg != 17)
        # if not np.any(mask):
        #     continue
        mask = (sseg > -1)


        # 相机点云和像素坐标
        points_cam, u, v = publish3D_rviz.depth_to_pointcloud(depth, mask)
        points_robot = points_cam @ R_robot_to_cam
        points_local = publish3D_rviz.transform_pointcloud_to_firstFrameCoordinate(points_robot, [0, 0, 0])
        local3D = points_local[np.newaxis, ...]  # shape: [1, N, 3]
        local3D = torch.from_numpy(local3D).float().to("cuda")

        points2D = np.stack((u, v), axis=-1)[np.newaxis, ...]  # shape: [1, N, 2]
        points2D = torch.from_numpy(points2D).float().to("cuda")

        ssegs_3 = sseg[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        ssegs_3 = torch.from_numpy(ssegs_3).float().to("cuda")

        # 构建语义栅格
        ego_semantic_sseg_27 = map_utils.ground_projection_my(
            points2D, local3D, ssegs_3,
            sseg_labels=object_labels,
            grid_dim=grid_dim,
            cell_size=cell_size
        )  # shape: [t, 27, 184, 184]
        print("ego_grid_sseg_3.shape: ", ego_semantic_sseg_27.shape)

        # 累加到全局地图
        # agent_pose, y_height = utils.get_sim_location( agent_state=self.test_ds.sim.get_agent_state())  # pose = x, y, yaw
        #      [zhjd-debug] self.grid_dim:  (384, 384)
        #      [zhjd-debug] self.cell_size:  0.1
        #      [zhjd-debug] ego_grid_sseg_3 size:  torch.Size([1, 3, 384, 384])
        #      [zhjd-debug] 裁切后 step_ego_grid_crops size:  torch.Size([1, 3, 64, 64])
        #      [zhjd-debug] agent_pose:  (1.8564888, 3.858087, -2.4431847146327996)
        #      [zhjd-debug] y_height:  0.09859601
        #     [zhjd-debug] rel:  (-1.6411479, 0.23411036, -1.3962633868516523)
        #     [zhjd-debug] _rel_pose:  tensor([[-1.6411,  0.2341, -1.3963]])
        abs_poses.append(position)
        # rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
        _rel_pose = torch.Tensor(position).unsqueeze(0).float()
        _rel_pose = _rel_pose.to(device)
        # global_grid += ego_grid_sseg_3
        geo_semantic_sseg = sg.mapTransformer(grid=ego_semantic_sseg_27, pose=_rel_pose,
                                              abs_pose=torch.tensor(abs_poses).to(device))
        step_geo_grid_sseg = sg.update_semantic_proj_grid_bayes(geo_grid=geo_semantic_sseg.unsqueeze(0))

        # 从世界坐标系转移到机器人坐标系
        step_ego_grid_sseg = sg.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=_rel_pose,
                                           abs_pose=torch.tensor(abs_poses).to(device))

        # Crop the grid around the agent at each timestep
        # # 剪切
        step_ego_grid_crops = map_utils.crop_grid(grid=step_ego_grid_sseg, crop_size=crop_size)
        step_ego_grid_crops = step_ego_grid_crops.squeeze(0)
        print("     [zhjd-debug] 裁切后 step_ego_grid_crops size: ", step_ego_grid_crops.size())


        # 多帧融合后取 argmax 得到最终语义图
        flag = 2

        if(flag == 1):
            ego_semantic_sseg_27 = ego_semantic_sseg_27.squeeze(0)
            label_map = ego_semantic_sseg_27.argmax(dim=0).cpu().numpy()  # shape: (H, W)
        if (flag == 11):
            geo_semantic_sseg = geo_semantic_sseg.squeeze(0)
            label_map = geo_semantic_sseg.argmax(dim=0).cpu().numpy()  # shape: (H, W)
        if(flag == 2):
            # print("step_geo_grid_sseg.shape: ", step_geo_grid_sseg.shape)
            step_ego_grid_sseg = step_ego_grid_sseg.squeeze(0).squeeze(0)
            # print("step_geo_grid_sseg.shape: ", step_geo_grid_sseg.shape)
            label_map = step_ego_grid_sseg.argmax(dim=0).cpu().numpy()  # shape: (H, W)
        if(flag == 3):
            step_ego_grid_crops = step_ego_grid_crops.squeeze(0)
            label_map = step_ego_grid_crops.argmax(dim=0).cpu().numpy()  # shape: (H, W)


        print("label_map.shape: ", label_map.shape)

        ego_vis = viz_utils.colorEncode(label_map)

        # 可视化融合地图
        plt.figure(figsize=(10, 10))
        plt.imshow(ego_vis)
        plt.title("Fused Semantic Map (10 Frames)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()