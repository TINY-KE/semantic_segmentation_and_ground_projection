#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ROSImage
import cv2
import message_filters  # --- 必须引入：用于消息同步
import tf2_ros         # --- 用于处理 TF
import tf2_geometry_msgs
import tf.transformations  # 确保导入此库以进行四元数转换

# System libs
import sys
import os
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time # 在文件开头添加 import time
import argparse
import threading  # 新增：线程库
import queue      # 新增：队列（线程安全）
# Numerical libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from mit_semseg.lib.utils import as_numpy
from PIL import Image as PILImage # 别名，解决命名冲突
from mit_semseg.config import cfg
from ground_projection.util import Id_Converter


from ground_projection.util.semantic_grid import SemanticGrid
from ground_projection.util import viz_utils, map_utils, utils, load_slam_dataset
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import rospy
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from ground_projection.publish3D_util.publish3D_rviz import publish3D
from ground_projection.publish3D_util.SemanticMapPublisher import SemanticMarkerPublisher, AsyncSemanticMarkerPublisher
from visualization_msgs.msg import Marker
from StepEgoMapPose_msgs.msg import StepEgoMapPose # 确保你已经编译了此消息包
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from PIL import Image as PILImage
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import shutil

    full_image_path = "/home/robotlab/work2/semantic-segmentation-pytorch/save_results/ros_segmentation_online_7floor/combined_image.png" 

    # 4. 调用逆向读取函数，从本地 PNG 恢复 Tensor
    print("\n[阶段 3] 执行读取操作 (load_Global_fromROS)...")
    recovered_tensor = viz_utils.load_Global_fromROS(full_image_path, color_mapping=viz_utils.color_mapping_27)

    # 5. 验证闭环结果
    print("\n[阶段 4] 结果验证...")
    print(f"恢复后的 Tensor 形状是否正确: {recovered_tensor.shape}")
    
    # 提取恢复后的 Tensor 中存在的类别 ID
    # 由于恢复后是 One-Hot 编码，我们在 channel 维度上取 argmax 即可得到类别图
    recovered_label_map = torch.argmax(recovered_tensor[0, 0], dim=0).numpy()
    unique_classes_recovered = np.unique(recovered_label_map)
    
    rospy.loginfo("[阶段 3] 正在生成可视化验证窗口，请查看弹出的 Matplotlib 窗口...")
    plt.figure(figsize=(12, 6))
    plt.title("Recovered Semantic Label Map (Argmax Result)")
    # 使用 tab20 等离散 colormap 可以更清晰地区分不同的 class ID
    im = plt.imshow(recovered_label_map, cmap='tab20', interpolation='nearest')
    plt.colorbar(im, label='Semantic Class ID')
    plt.axis('on')
    plt.show()