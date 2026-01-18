#!/usr/bin/env python
# coding: utf-8

import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# # 图像尺寸与相机内参
# Camera.fx: 481.20
# Camera.fy: 480.00
# Camera.cx: 319.50
# Camera.cy: 239.50
# # camera
# Camera.width: 640
# Camera.height: 480
# # Deptmap values factor
# DepthMapFactor: 5000  #对应
W = 640
H = 480
fx = 481.20
fy = -480.00
cx = 319.50
cy = 239.50
DepthMapFactor = 5000.0

# 颜色映射（类别 ID → RGB）
color_mapping_27 = {
    0:  (255, 255, 255),   # white
    1:  (128, 128, 0),     # olive
    2:  (0, 0, 255),       # blue
    3:  (255, 0, 0),       # red
    4:  (255, 0, 255),     # magenta
    5:  (0, 255, 255),     # cyan
    6:  (255, 165, 0),     # orange
    7:  (255, 255, 0),     # yellow
    8:  (128, 128, 128),   # gray
    9:  (128, 0, 0),       # maroon
    10: (255, 20, 147),    # deep pink
    11: (0, 128, 0),       # dark green
    12: (128, 0, 128),     # purple
    13: (0, 128, 128),     # teal
    14: (0, 0, 128),       # navy
    15: (210, 105, 30),    # chocolate
    16: (188, 143, 143),   # rosy brown
    17: (0, 255, 0),       # green
    18: (255, 215, 0),     # gold
    19: (0, 0, 0),         # black
    20: (192, 192, 192),   # silver
    21: (138, 43, 226),    # blue violet
    22: (255, 127, 80),    # coral
    23: (238, 130, 238),   # violet
    24: (245, 245, 220),   # beige
    25: (139, 69, 19),     # saddle brown
    26: (64, 224, 208)     # turquoise
}

def depth_to_pointcloud(depth, mask):
    """将深度图中的选中像素转换为相机坐标系下的 3D 点云"""
    h, w = depth.shape
    print("h:{}, w:{}".format(h, w))
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u[mask]
    v = v[mask]
    z = depth[mask]/DepthMapFactor
    # ✅ 过滤掉 z < 1 的点
    valid = z >= 1
    u = u[valid]
    v = v[valid]
    z = z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack((x, y, z), axis=-1)  # (N, 3)

def transform_pointcloud_to_world(points, camera_position):
    """将点云从机器人坐标系变换到世界坐标系"""
    translation = np.array(camera_position[:3])  # 平移向量: [x, y, z]
    quaternion = np.array(camera_position[3:])  # 四元数: [qx, qy, qz, qw]

    # 四元数 → 旋转矩阵
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat(quaternion)
    rotation_matrix = rot.as_matrix()  # shape: (3, 3)

    # 应用旋转和平移: world_point = R * point + t
    rotated_points = points @ rotation_matrix.T  # shape: (N, 3)
    world_points = rotated_points + translation  # 平移

    return world_points  # shape: (N, 3)

def filter_depth_edges(depth, threshold=0.2):
    """过滤深度图边缘的异常跳变（高梯度）"""
    depth = depth.astype(np.float32)
    dz_x = np.abs(np.diff(depth, axis=1, append=depth[:, -1:]))
    dz_y = np.abs(np.diff(depth, axis=0, append=depth[-1:, :]))
    edge_mask = (dz_x < threshold) & (dz_y < threshold)
    return edge_mask


def publish_marker_pointcloud(points, marker_id=0, color=(0.0, 1.0, 0.0), scale=0.05):
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
        pt = Point()
        pt.x = float(x)
        pt.y = float(y)
        pt.z = float(z)
        marker.points.append(pt)

    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node("semantic_pointcloud_publisher")
    marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
    rospy.sleep(1.0)  # 等待初始化

    # 加载数据
    npz_file_path = "/home/robotlab/work/semantic-segmentation-pytorch/save_results/all_data.npz"
    data = np.load(npz_file_path)
    all_sseg = data["ssegs"]       # (N, H, W)
    all_depth = data["depth_imgs"] # (N, H, W)
    all_pose = data["abs_pose"]    # (N, 7) 每行分别为tx ty tz qx qy qz qw

    for i in range(len(all_sseg)):
        depth = np.squeeze(all_depth[i])
        sseg = np.squeeze(all_sseg[i])
        camera_position = np.squeeze(all_pose[i])  #tx ty tz qx qy qz qw

        rospy.loginfo("帧 %d，位姿: %s", i, camera_position)

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

        mask = (sseg > -1)
        edge_mask = filter_depth_edges(depth, threshold=0.1)

        # 遍历所有类别
        for class_id in range(27):
            mask = (sseg == class_id)
            if not np.any(mask):
                continue

            final_mask = mask & edge_mask

            points_cam = depth_to_pointcloud(depth, final_mask)
            points_in_first_camera_pose = transform_pointcloud_to_world(points_cam, camera_position)
            points_world = points_in_first_camera_pose@First_Camera_Pose_ICL + First_Camera_t

            # 可选过滤：仅保留所有坐标小于 2 的点
            mask_filter = points_world[:, 2] < 2
            points_world = points_world[mask_filter]

            if points_world.shape[0] == 0:
                continue

            rgb = color_mapping_27.get(class_id, (255, 255, 255))
            color = tuple([c / 255.0 for c in rgb])
            marker_id = i * 100 + class_id  # 保证唯一 ID

            publish_marker_pointcloud(points_world, marker_id=marker_id, color=color, scale=0.01)
            # publish_marker_pointcloud(points_world, marker_id=marker_id, color=color, scale=0.003)

        # mask = (sseg>-1)
        # points_cam = depth_to_pointcloud(depth, mask)
        # print("points_cam size: ",points_cam.size)  # points_cam size:  921600
        # points_world = transform_pointcloud_to_world(points_cam, camera_position)
        # publish_marker_pointcloud(points_world, marker_id=1, color= (255, 0, 2), scale=0.3)

        rospy.sleep(0.5)  # 每帧间隔