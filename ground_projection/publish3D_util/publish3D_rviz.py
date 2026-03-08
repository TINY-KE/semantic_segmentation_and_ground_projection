#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R
from PIL import Image
import cv2
from ground_projection.util import camera_intrinsics
from std_msgs.msg import ColorRGBA

# 语义颜色映射（27类）
color_mapping_27 = {
    0:  (255, 255, 255),   1:  (128, 128, 0),     2:  (0, 0, 255),
    3:  (255, 0, 0),       4:  (255, 0, 255),     5:  (0, 255, 255),
    6:  (255, 165, 0),     7:  (255, 255, 0),     8:  (128, 128, 128),
    9:  (128, 0, 0),       10: (255, 20, 147),    11: (0, 128, 0),
    12: (128, 0, 128),     13: (0, 128, 128),     14: (0, 0, 128),
    15: (210, 105, 30),    16: (188, 143, 143),   17: (0, 255, 0),
    18: (255, 215, 0),     19: (0, 0, 0),         20: (192, 192, 192),
    21: (138, 43, 226),    22: (255, 127, 80),    23: (238, 130, 238),
    24: (245, 245, 220),   25: (139, 69, 19),     26: (64, 224, 208)
}

class publish3D:
    def __init__(self, camera_type="KINECT_DK"):
        # self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
        self.W, self.H, self.fx, self.fy, self.cx, self.cy, self.DepthMapFactor = camera_intrinsics.get_intrinsics("KINECT_DK")

    def filter_depth_edges(self, depth, threshold=0.2):
        """过滤深度图边缘的异常跳变（高梯度）"""
        depth = depth.astype(np.float32)
        dz_x = np.abs(np.diff(depth, axis=1, append=depth[:, -1:]))
        dz_y = np.abs(np.diff(depth, axis=0, append=depth[-1:, :]))
        edge_mask = (dz_x < threshold) & (dz_y < threshold)
        return edge_mask



    def filter_depth_range(self, depth: np.ndarray, min_depth=0.2, max_depth=5.0) -> np.ndarray:
        """
        过滤深度图中不在指定范围内的像素。

        参数:
            depth (np.ndarray): 输入深度图（单位为米），shape 为 (H, W)
            min_depth (float): 最小有效深度值（默认 0.2 米）
            max_depth (float): 最大有效深度值（默认 5.0 米）

        返回:
            mask (np.ndarray): 布尔型掩码，True 表示该像素深度有效
        """
        mask = (depth >= min_depth) & (depth <= max_depth)
        return mask


    def depth_voxel_mask(self, depth: np.ndarray, voxel_size: int = 4) -> np.ndarray:
        """
        对深度图做体素风格降采样，返回掩码（mask）。

        每个 voxel_size x voxel_size 的区域只保留一个像素（左上）。

        参数:
            depth (np.ndarray): 输入深度图，shape (H, W)
            voxel_size (int): 网格大小（单位：像素）

        返回:
            mask (np.ndarray): 布尔数组，True 表示该像素被保留
        """
        h, w = depth.shape
        mask = np.zeros_like(depth, dtype=bool)

        for y in range(0, h, voxel_size):
            for x in range(0, w, voxel_size):
                if depth[y, x] > 0:  # 只保留有效深度
                    mask[y, x] = True

        return mask



    def depth_to_pointcloud(self, depth, mask):
        edge_mask = self.filter_depth_edges(depth, threshold=0.2)
        final_mask = mask & edge_mask

        """将深度图中的选中像素转换为相机坐标系下的 3D 点云，并统计被过滤掉的点数量"""
        h, w = depth.shape
        # print("h:{}, w:{}".format(h, w))
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u[final_mask]
        v = v[final_mask]
        z = depth[final_mask] / self.DepthMapFactor
        # ✅ 过滤条件
        valid = z >= 1
        # 只保留 valid 的点
        u = u[valid]
        v = v[valid]
        z = z[valid]

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return np.stack((x, y, z), axis=-1), u, v  # shape: (N, 3)


    def depth_to_pointcloud_camera(self, depth):
        edge_mask = self.filter_depth_edges(depth, threshold=0.2)
        voxel_mask = self.depth_voxel_mask(depth, voxel_size=4) # 每两个像素保留一个
        final_mask = voxel_mask

        """将深度图中的选中像素转换为相机坐标系下的 3D 点云，并统计被过滤掉的点数量"""
        h, w = depth.shape
        # print("h:{}, w:{}".format(h, w))
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u[final_mask]
        v = v[final_mask]
        z = depth[final_mask] / self.DepthMapFactor
        # # ✅ 过滤条件
        # valid = z >= 1
        # # 只保留 valid 的点
        # u = u[valid]
        # v = v[valid]
        # z = z[valid]

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return np.stack((x, y, z), axis=-1), u, v

    def transform_pointcloud_to_world(self, points, rot, trans):
        points_world = points @  rot.T + trans
        return points_world



    def transform_pointcloud_to_firstFrameCoordinate(self, points, camera_position):
        """将点云从相机坐标系变换到世界坐标系"""
        translation = np.array(camera_position[:3])   # 平移向量: tx, ty, tz
        quaternion = np.array(camera_position[3:])    # 四元数: qx, qy, qz, qw
        print(f">>> Pose Info:")
        print(f"    Translation (xyz): {translation}")
        print(f"    Quaternion (xyzw): {quaternion}")
        rot = R.from_quat(quaternion)
        rotation_matrix = rot.as_matrix()
        rotated_points = points @ rotation_matrix.T
        points_firstFrameCoordinate = rotated_points + translation
        return points_firstFrameCoordinate

    def publish_marker_pointcloud(self, marker_pub, points, marker_id=0, color=(0.0, 1.0, 0.0), scale=0.05):
        """使用 visualization_msgs/Marker 发布点云为小球列表"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "semantic_points"
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        for x, y, z in points:
            pt = Point(x=float(x), y=float(y), z=float(z))
            marker.points.append(pt)

        marker.lifetime = rospy.Duration(0)
        marker_pub.publish(marker)

    def Publish3D(self):
        rospy.init_node("semantic_pointcloud_publisher")
        global marker_pub
        marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
        rospy.sleep(1.0)

        npz_file_path = "/home/robotlab/work/semantic-segmentation-pytorch/save_results/all_data.npz"
        data = np.load(npz_file_path)
        all_sseg = data["ssegs"]          # (N, H, W)
        all_depth = data["depth_imgs"]    # (N, H, W)
        all_pose = data["abs_pose"]       # (N, 7)

        for i in range(len(all_sseg)):
            depth = np.squeeze(all_depth[i])
            sseg = np.squeeze(all_sseg[i])
            camera_position = np.squeeze(all_pose[i])

            rospy.loginfo("帧 %d，位姿: %s", i, camera_position)

            for class_id in range(27):
                mask = (sseg == class_id)
                if not np.any(mask):
                    continue

                points_cam, u, v = self.depth_to_pointcloud(depth, mask)
                if points_cam.size == 0:
                    continue

                points_world = self.transform_pointcloud_to_firstFrameCoordinate(points_cam, camera_position)

                if points_world.shape[0] == 0:
                    continue

                rgb = color_mapping_27.get(class_id, (255, 255, 255))
                color = tuple([c / 255.0 for c in rgb])
                marker_id = i * 100 + class_id

                self.publish_marker_pointcloud(marker_pub, points_world, marker_id=marker_id, color=color, scale=0.003)

            rospy.sleep(0.5)


    def publish3D_from_depth_path(self, marker_pub, depth_path, pose, default_marker_id):
        # 读取depth
        depth_img = Image.open(depth_path)
        depth_np = np.array(depth_img).astype(np.float32)  # 深度图原始值 (H, W)
        # 构建 mask：非零深度为有效点
        mask = (depth_np > 0).astype(np.uint8)
        points_cam, _, _ = self.depth_to_pointcloud_camera(depth_np)
        print("zhjd-debug, points_cam size: ", points_cam.shape)
        points_world = self.transform_pointcloud_to_firstFrameCoordinate(points_cam, pose)
        print("发布到RVIZ")
        self.publish_marker_pointcloud(marker_pub, points_world, marker_id=default_marker_id, color=(0, 2, 255), scale=0.03)
        rospy.sleep(0.5)  # 每帧间隔

    def publish_marker_pointcloud_with_color(self, marker_pub, points, colors, marker_id, scale):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = marker_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # 设置点的大小
        marker.scale.x = scale
        marker.scale.y = scale

        # 这里的颜色必须设为1.0（或者不设置），否则整体会被叠加一层颜色
        marker.color.a = 1.0

        for i in range(len(points)):
            p = Point()
            p.x, p.y, p.z = points[i]
            marker.points.append(p)

            c = ColorRGBA()
            # 确保颜色通道正确分配
            c.r = float(colors[i][0])
            c.g = float(colors[i][1])
            c.b = float(colors[i][2])
            c.a = 1.0
            marker.colors.append(c)

        marker_pub.publish(marker)

    def publish3D_from_depth_rgb_path(self, marker_pub, depth_path, rgb_path, pose, default_marker_id):
        # 1. 读取深度图
        depth_img = Image.open(depth_path)
        depth_np = np.array(depth_img).astype(np.float32)

        # 2. 读取 RGB 图像
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_np = np.array(rgb_img)

        # 3. 获取坐标点及对应的像素索引 u, v
        points_cam, u, v = self.depth_to_pointcloud_camera(depth_np)

        if points_cam.size == 0:
            return

        # 4. 坐标系转换：从相机坐标系转到世界/第一帧坐标系
        points_world = self.transform_pointcloud_to_firstFrameCoordinate(points_cam, pose)

        # --- 核心新增：高度过滤逻辑 ---
        # 在 ROS 标准坐标系中，Z 通常代表高度（Up 方向）
        # 创建一个布尔掩码，只保留高度 <= 3.0 米的点
        height_threshold = 2.3
        height_mask = points_world[:, 2] <= height_threshold

        # 应用掩码过滤坐标
        points_filtered = points_world[height_mask]

        # 如果所有点都被过滤掉了，直接返回
        if points_filtered.size == 0:
            return
        # -----------------------------

        # 5. 提取并过滤对应的颜色
        # 确保颜色提取与坐标点完全对应
        # 注意：此处需要先用 v, u 索引提取全量颜色，再应用高度掩码
        colors_all = rgb_np[v, u] / 255.0
        colors_filtered = colors_all[height_mask]

        # 6. 发布过滤后的彩色点云
        self.publish_marker_pointcloud_with_color(
            marker_pub,
            points_filtered,
            colors_filtered,
            marker_id=default_marker_id,
            scale=0.01
        )


if __name__ == "__main__":
    try:
        vis = publish3D("KINECT_DK")
        vis.Publish3D()
    except rospy.ROSInterruptException:
        pass