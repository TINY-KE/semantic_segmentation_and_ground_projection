#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R
from PIL import Image

# 相机内参和深度缩放因子
W = 640
H = 480
fx = 481.20
fy = -480.00
cx = 319.50
cy = 239.50
DepthMapFactor = 5000.0

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

def filter_depth_edges(depth, threshold=0.2):
    """过滤深度图边缘的异常跳变（高梯度）"""
    depth = depth.astype(np.float32)
    dz_x = np.abs(np.diff(depth, axis=1, append=depth[:, -1:]))
    dz_y = np.abs(np.diff(depth, axis=0, append=depth[-1:, :]))
    edge_mask = (dz_x < threshold) & (dz_y < threshold)
    return edge_mask


def depth_to_pointcloud(depth, mask):
    edge_mask = filter_depth_edges(depth, threshold=0.2)
    final_mask = mask & edge_mask

    """将深度图中的选中像素转换为相机坐标系下的 3D 点云，并统计被过滤掉的点数量"""
    h, w = depth.shape
    # print("h:{}, w:{}".format(h, w))
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u[final_mask]
    v = v[final_mask]
    z = depth[final_mask] / DepthMapFactor
    # ✅ 过滤条件
    valid = z >= 1
    # 只保留 valid 的点
    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack((x, y, z), axis=-1), u, v  # shape: (N, 3)


def transform_pointcloud_to_world(points, rot, trans):
    points_world = points @  rot.T + trans
    return points_world



def transform_pointcloud_to_firstFrameCoordinate(points, camera_position):
    """将点云从相机坐标系变换到世界坐标系"""
    translation = np.array(camera_position[:3])   # 平移向量: tx, ty, tz
    quaternion = np.array(camera_position[3:])    # 四元数: qx, qy, qz, qw
    rot = R.from_quat(quaternion)
    rotation_matrix = rot.as_matrix()
    rotated_points = points @ rotation_matrix.T
    points_firstFrameCoordinate = rotated_points + translation
    return points_firstFrameCoordinate

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
        pt = Point(x=float(x), y=float(y), z=float(z))
        marker.points.append(pt)

    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)

def Publish3D():
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

            points_cam = depth_to_pointcloud(depth, mask)
            if points_cam.size == 0:
                continue

            points_world = transform_pointcloud_to_firstFrameCoordinate(points_cam, camera_position)

            if points_world.shape[0] == 0:
                continue

            rgb = color_mapping_27.get(class_id, (255, 255, 255))
            color = tuple([c / 255.0 for c in rgb])
            marker_id = i * 100 + class_id

            publish_marker_pointcloud(points_world, marker_id=marker_id, color=color, scale=0.003)

        rospy.sleep(0.5)

def publish3D_from_depth_path(depth_path, pose):
    # 读取depth
    depth_img = Image.open(depth_path)
    depth_np = np.array(depth_img).astype(np.float32)  # 深度图原始值 (H, W)
    points_cam = depth_to_pointcloud(depth_np)
    print("zhjd-debug, points_cam size: ", points_cam.shape)
    points_world = transform_pointcloud_to_firstFrameCoordinate(points_cam, pose)
    default_marker_id = 0
    default_marker_id += 1
    publish_marker_pointcloud(points_world, marker_id=default_marker_id, color=(0, 2, 255), scale=0.03)
    rospy.sleep(0.5)  # 每帧间隔


if __name__ == "__main__":
    try:
        Publish3D()
    except rospy.ROSInterruptException:
        pass