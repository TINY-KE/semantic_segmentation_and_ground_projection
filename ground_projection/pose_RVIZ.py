#!/usr/bin/env python
# coding: utf-8

import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':
    rospy.init_node("semantic_pointcloud_publisher")
    marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
    rospy.sleep(1.0)  # 等待发布器初始化

    # 加载数据
    npz_file_path = "/home/robotlab/work/semantic-segmentation-pytorch/save_results/all_data.npz"
    data = np.load(npz_file_path)
    all_sseg = data["ssegs"]       # (N, H, W)
    all_depth = data["depth_imgs"] # (N, H, W)
    all_pose = data["abs_pose"]    # (N, 7): tx, ty, tz, qx, qy, qz, qw

    # 相机坐标系 → 世界坐标系旋转矩阵（ICL）
    First_Camera_Rot_ICL = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    for i in range(len(all_sseg)):
        depth = np.squeeze(all_depth[i])
        sseg = np.squeeze(all_sseg[i])
        camera_position = np.squeeze(all_pose[i])  # [tx, ty, tz, qx, qy, qz, qw]

        rospy.loginfo("帧 %d，位姿: %s", i, camera_position)

        # 分离平移和四元数
        translation = np.array(camera_position[:3])
        quaternion = np.array(camera_position[3:])

        # 四元数 → 旋转矩阵
        rot = R.from_quat(quaternion)
        rotation_matrix = rot.as_matrix()

        # 坐标变换：相机 → 世界
        rot_world = First_Camera_Rot_ICL.T @ rotation_matrix
        trans_world = First_Camera_Rot_ICL.T @ translation  # shape: (3,)

        ## ---------------- 创建 Marker (1): 位置球 ---------------- ##
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_positions"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = trans_world[0]
        marker.pose.position.y = trans_world[1]
        marker.pose.position.z = trans_world[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        # marker.lifetime = rospy.Duration(5.0)
        marker_pub.publish(marker)

        ## ---------------- 创建 Marker (2): 朝向箭头 ---------------- ##
        # 相机前向向量：Z轴
        camera_forward_cam = np.array([0, 0, 1])
        camera_forward_world = rot_world @ camera_forward_cam

        arrow_length = 0.5
        arrow_tip = trans_world + camera_forward_world * arrow_length

        arrow_marker = Marker()
        arrow_marker.header.frame_id = "map"
        arrow_marker.header.stamp = rospy.Time.now()
        arrow_marker.ns = "camera_orientations"
        arrow_marker.id = i + 1000
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.scale.x = 0.05  # 箭杆粗细
        arrow_marker.scale.y = 0.1   # 箭头宽度
        arrow_marker.scale.z = 0.1   # 箭头高度
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 1.0
        # arrow_marker.lifetime = rospy.Duration(5.0)

        start_point = Point(*trans_world)
        end_point = Point(*arrow_tip)
        arrow_marker.points = [start_point, end_point]
        marker_pub.publish(arrow_marker)

        ## ---------------- 等待下一帧 ---------------- ##
        rospy.sleep(0.5)