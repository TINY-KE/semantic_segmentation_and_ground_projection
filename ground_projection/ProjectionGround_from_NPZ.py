#!/usr/bin/env python
# coding: utf-8
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ground_projection.util.semantic_grid import SemanticGrid
from ground_projection.util import viz_utils, map_utils, utils, load_slam_dataset
from ground_projection.publish3D_util import publish3D_rviz
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import rospy
import sensor_msgs.point_cloud2 as pc2

def create_pointcloud2_from_xyz(points, frame_id="map"):
    """
    将 Nx3 的 NumPy 点云数组转换为 ROS 的 PointCloud2 消息
    """
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]

    pc_data = [tuple(p) for p in points]
    cloud_msg = pc2.create_cloud(header, fields, pc_data)
    return cloud_msg


if __name__ == '__main__':
    rospy.init_node("semantic_projection_publisher")
    pc_pub = rospy.Publisher("/pointcloud_world", PointCloud2, queue_size=1)
    rospy.sleep(1.0)

    # 一、 加载数据
    npz_file_path = "/home/robotlab/work/semantic-segmentation-pytorch/save_results/all_data.npz"
    data = np.load(npz_file_path)
    all_sseg = data["ssegs"]       # (N, H, W)
    all_depth = data["depth_imgs"] # (N, H, W)
    all_pose = data["abs_pose"]    # (N, 7)
    all_images = data["images"]
    virtual_robot_ground_poses = []
    ego_crops = []

    # 地图参数
    spatial_labels = 3
    object_labels = 27
    grid_dim = (400, 400)
    # grid_dim = (64, 64)
    cell_size = 0.1
    crop_size = (64, 64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 二、初始化全局地图张量
    global_grid = torch.zeros((1, spatial_labels, *grid_dim), dtype=torch.float32).to("cuda")
    # For each episode we need a new instance of a fresh global grid
    sg = SemanticGrid(1, grid_dim, crop_size[0], cell_size,
                      spatial_labels=spatial_labels, object_labels=object_labels)

    # 三、 读取每一个时间步骤的信息
    for i in range(len(all_sseg)):
        depth = np.squeeze(all_depth[i])
        sseg = np.squeeze(all_sseg[i])
        camera_position = np.squeeze(all_pose[i])  # tx ty tz qx qy qz qw

        # 四、计算相机在world坐标系下的pose
        # 相机 → 机器人坐标系变换
        # 机器人坐标系：x轴朝前，y轴朝左，z轴朝上，
        #  y轴朝左，x轴朝下，z轴朝前
        First_Camera_Rot = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        # ICL： y轴朝上，x轴朝左，z轴朝前    相机坐标系 =  First_Camera_Rot_ICL * 机器人坐标系
        First_Camera_Rot_ICL = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        first_init_flag = False
        first_init_pose = None

        First_Camera_t = np.array([-2.25, 0, 1.17])

        # mask = (sseg != 17)
        # if not np.any(mask):
        #     continue
        mask = (sseg > -1)

        print("帧 {}，位姿: {}".format(i, camera_position))

        # 分离平移和四元数
        translation = np.array(camera_position[:3])
        quaternion = np.array(camera_position[3:])


        # 四元数 → 旋转矩阵
        rot = R.from_quat(quaternion)
        rotation_matrix = rot.as_matrix()

        # 坐标变换：相机 → 世界
        rot_world = First_Camera_Rot_ICL.T @ rotation_matrix
        trans_world = First_Camera_Rot_ICL.T @ translation  + [0,0,1.17] # shape: (3,)
        if not first_init_flag:
            first_init_pose = trans_world
            first_init_flag = True

        # 五、计算点云在相机坐标系下的坐标
        # 相机点云和像素坐标
        points_cam, u, v = publish3D_rviz.depth_to_pointcloud(depth, mask)
        points_world = publish3D_rviz.transform_pointcloud_to_world(points_cam, rot_world, trans_world)

        # 发布点云到 RViz， 用于检查
        # if points_world.shape[0] > 10:
        #     pc_msg = create_pointcloud2_from_xyz(points_world, frame_id="map")
        #     pc_pub.publish(pc_msg)

        local3D = points_world[np.newaxis, ...]  # shape: [1, N, 3]
        local3D = torch.from_numpy(local3D).float().to("cuda")

        points2D = np.stack((u, v), axis=-1)[np.newaxis, ...]  # shape: [1, N, 2]
        points2D = torch.from_numpy(points2D).float().to("cuda")

        # 有效点云对应的语义分割像素点
        ssegs_3 = sseg[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        ssegs_3 = torch.from_numpy(ssegs_3).float().to("cuda")

        # 六、地面投影，构建单帧语义栅格地图
        ego_semantic_sseg_27 = map_utils.ground_projection_my(
            points2D, local3D, ssegs_3,
            sseg_labels=object_labels,
            grid_dim=grid_dim,
            cell_size=cell_size
        )  # shape: [t, 27, 184, 184]
        # print("ego_grid_sseg_3.shape: ", ego_semantic_sseg_27.shape)


        # 七、累加到全局地图
        geo_semantic_sseg = ego_semantic_sseg_27
        step_geo_grid_sseg = sg.update_semantic_proj_grid_bayes(geo_grid=geo_semantic_sseg.unsqueeze(0))

        # 八、计算【虚拟机器人平台】在world坐标系下的平面坐标virtual_robot_ground_pose
        first_z_world = np.array([1, 0, 0])
        # 相机 Z 轴方向（本体坐标系）
        z_cam = np.array([0, 0, 1])

        # 世界坐标系下相机朝向（Z 轴）
        z_world = rot_world @ z_cam  # shape: (3,)

        # 投影到水平面 (XY)
        z_proj = z_world[:2]  # 取 x 和 y 分量
        norm = np.linalg.norm(z_proj)
        # 如果投影为零向量，跳过
        if np.linalg.norm(z_proj) < 1e-6:
            angle_deg = 0.0
        else:
            # 单位化
            a = z_proj[:2] / np.linalg.norm(z_proj[:2])  # 当前方向
            b = first_z_world[:2] / np.linalg.norm(first_z_world[:2])  # 参考方向

            # 点积求夹角
            dot = np.clip(np.dot(a, b), -1.0, 1.0)
            angle_rad = np.arccos(dot)
            angle_deg = np.degrees(angle_rad)

            # 用 2D 叉积判断方向（正：左转，负：右转）
            cross = a[0] * b[1] - a[1] * b[0]
            if cross < 0:
                angle_deg = -angle_deg
            angle_rad = angle_deg / 180 * np.pi

        # print(f"相机朝向与X轴夹角：{abs(angle_deg):.2f}°")

        print("机器人位姿： ", [trans_world[0], trans_world[1],  angle_deg])
        virtual_robot_ground_pose = [-1* trans_world[0], -1* trans_world[1],  angle_rad]
        virtual_robot_ground_poses.append(virtual_robot_ground_pose)


        # 九、将全局地图转移到机器人为中心【ego】
        step_ego_grid_sseg = sg.transform_global_to_ego_single(grid=step_geo_grid_sseg.squeeze(0).squeeze(0), abs_pose=torch.tensor(virtual_robot_ground_pose).to(device))
        # Crop the grid around the agent at each timestep

        # 十、剪切
        step_ego_grid_crops = map_utils.crop_grid(grid=step_ego_grid_sseg, crop_size=crop_size)
        step_ego_grid_crops = step_ego_grid_crops.squeeze(0)
        ego_crops.append(step_ego_grid_crops)

        # 十一、可视化和保存到本地
        flag = 4
        if(flag == 1):
            ego_semantic_sseg_27 = ego_semantic_sseg_27.squeeze(0)
            label_map = ego_semantic_sseg_27.argmax(dim=0).cpu().numpy()  # shape: (H, W)
        if (flag == 2):
            step_geo_grid_sseg = step_geo_grid_sseg.squeeze(0).squeeze(0)
            label_map = step_geo_grid_sseg.argmax(dim=0).cpu().numpy()  # shape: (H, W)
        if (flag == 3):
            # print("step_geo_grid_sseg.shape: ", step_geo_grid_sseg.shape)
            step_ego_grid_sseg = step_ego_grid_sseg.squeeze(0).squeeze(0)
            # print("step_geo_grid_sseg.shape: ", step_geo_grid_sseg.shape)
            label_map = step_ego_grid_sseg.argmax(dim=0).cpu().numpy()  # shape: (H, W)
        if (flag == 4):
            step_ego_grid_crops = step_ego_grid_crops.squeeze(0)
            label_map = step_ego_grid_crops.argmax(dim=0).cpu().numpy()  # shape: (H, W)

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

        # 十二、添加到NPZ文件中
        # 检查all_sseg = data["ssegs"]       # (N, H, W)
        #     all_depth = data["depth_imgs"] # (N, H, W)
        #     all_pose = data["abs_pose"]    # (N, 7)
        #     virtual_robot_ground_poses = []
        #     ego_crops = []  长度是不是一致
        # === 循环之后，保存为 .npz 文件 ===
        virtual_robot_ground_poses_np = np.array(virtual_robot_ground_poses)  # (N, 3)
        ego_crops_tensor = torch.stack(ego_crops, dim=0)  # (N, C, H, W)
        ego_crops_np = ego_crops_tensor.cpu().numpy().astype(np.float32)  #概率值（通常在0-1之间）
        # #     # work2: ['abs_pose', 'ego_grid_crops_spatial', 'step_ego_grid_crops_spatial', 'gt_grid_crops_spatial', 'gt_grid_crops_objects',
        #     #     'images', 'ssegs', 'depth_imgs', 'pred_ego_crops_sseg', 'step_ego_grid_27']
        output_npz_path = "/home/robotlab/work/semantic-segmentation-pytorch/save_results/virtual_robot_outputs.npz"
        np.savez(
            output_npz_path,
            depth_imgs=all_depth,
            images=all_images,
            abs_pose=all_pose,
            ssegs=all_sseg,
            virtual_robot_ground_poses=virtual_robot_ground_poses_np,
            step_ego_grid_27=ego_crops_np
        )
        print("数据维度检查:")
        print("  depth_imgs.shape:", all_depth.shape)
        print("  images.shape:", all_images.shape)
        print("  all_pose.shape:", all_pose.shape)
        print("  all_sseg.shape:", all_sseg.shape)
        print("  virtual_robot_ground_poses.shape:", virtual_robot_ground_poses_np.shape)
        print("  step_ego_grid_27.shape:", ego_crops_np.shape)
