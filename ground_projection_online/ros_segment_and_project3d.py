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


# 你的全局变量保持不变
old_to_new_idx = Id_Converter.get_Id_Converter("binzhou_wjl")
DEFAULT_NEW_IDX = Id_Converter.DEFAULT_NEW_IDX
color_mapping_27 = {
    0: (255, 255, 255), 1: (128, 128, 0), 2: (0, 0, 255), 3: (255, 0, 0), 
    4: (255, 0, 255), 5: (0, 255, 255), 6: (255, 165, 0), 7: (255, 255, 0), 
    8: (128, 128, 128), 9: (128, 0, 0), 10: (255, 20, 147), 11: (0, 128, 0), 
    12: (128, 0, 128), 13: (0, 128, 128), 14: (0, 0, 128), 15: (210, 105, 30), 
    16: (188, 143, 143), 17: (0, 255, 0), 18: (255, 215, 0), 19: (0, 0, 0), 
    20: (192, 192, 192), 21: (138, 43, 226), 22: (255, 127, 80), 23: (238, 130, 238), 
    24: (245, 245, 220), 25: (139, 69, 19), 26: (64, 224, 208)
}
colors_27 = np.array([color_mapping_27[i] for i in sorted(color_mapping_27.keys())], dtype=np.uint8)




class ROSSegmentationNode:
    def __init__(self, cfg, gpu):
        self.cfg = cfg
        self.gpu = gpu
        self.bridge = CvBridge()
        self.is_busy = False  # --- 新增：忙碌状态锁 ---
        self.prev_time = time.time() # 初始化时间戳
        self.fps = 0.0

        # TF Buffer 初始化
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 构建3D投影模型
        self.pub3d = publish3D(camera_type="KINECT_DK_ROS")
        self.marker_pub = rospy.Publisher("/pointcloud_world", Marker, queue_size=10)

        # 构建语义分割模型
        self.net_encoder = ModelBuilder.build_encoder(arch=cfg.MODEL.arch_encoder, fc_dim=cfg.MODEL.fc_dim, weights=cfg.MODEL.weights_encoder)
        self.net_decoder = ModelBuilder.build_decoder(arch=cfg.MODEL.arch_decoder, fc_dim=cfg.MODEL.fc_dim, num_class=cfg.DATASET.num_class, weights=cfg.MODEL.weights_decoder, use_softmax=True)
        self.segmentation_module = SegmentationModule(self.net_encoder, self.net_decoder, None)
        self.segmentation_module.cuda(self.gpu)
        self.segmentation_module.eval()

        # 语义栅格地图构建参数
        self.spatial_labels = 3
        self.object_labels = 27
        self.grid_dim = (200, 200)
        self.cell_size = 0.1
        self.crop_size = (64, 64)
        self.sg = SemanticGrid(1, self.grid_dim, self.crop_size[0], self.cell_size,
                      spatial_labels=self.spatial_labels, object_labels=self.object_labels)

        self.semantic_map_publisher = AsyncSemanticMarkerPublisher(marker_topic="/semantic_global_map")

        # --- 预生成ID映射数组 ---
        # 找到old_to_new_idx中的最大旧ID（确定映射数组长度）
        max_old_id = max(old_to_new_idx.keys(), default=0)
        # 创建映射数组（索引=旧ID，值=新ID），默认值为DEFAULT_NEW_IDX
        self.id_mapping_array = np.full((max_old_id + 2,), DEFAULT_NEW_IDX, dtype=np.int32)
        # 填充映射关系
        for old_id, new_id in old_to_new_idx.items():
            self.id_mapping_array[old_id] = new_id
        # rospy.loginfo(f"预生成ID映射数组，长度: {len(self.id_mapping_array)}")


        #  
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        # 预定义旋转矩阵（只初始化一次）
        self.R_footprint_cam = torch.tensor([
            [0, 0, 1],   
            [-1, 0, 0],  
            [0, -1, 0]    
        ], dtype=torch.float32, device=self.device)

    def preprocess(self, img):
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1))
    
    def callback(self, rgb_msg, depth_msg):
        # --- 新增：只处理最新图像 ---
        if self.is_busy:
            return  # 如果上一帧还没处理完，直接跳过当前这一帧
        
        self.is_busy = True

        try:

            # 1. 获取 TF 数据 (示例：查询从 camera 到 base_link 的变换)
            try:
                # 获取 base_link 相对于 map 的变换
                transform = self.tf_buffer.lookup_transform(
                    "map",            # 目标坐标系 (Target Frame)
                    "base_footprint",      # 源坐标系 (Source Frame)
                    rgb_msg.header.stamp,  # 时间戳 (必须与图像时间戳对齐)
                    rospy.Duration(0.1)    # 超时时间
                )
                # 获取 x, y 坐标
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                # 获取四元数并转换为欧拉角
                quat = transform.transform.rotation
                # euler_from_quaternion 返回 (roll, pitch, yaw)
                euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
                theta = euler[2]  # yaw 角即为机器人当前的航向角 (单位: 弧度)
                # 打印或使用坐标
                rospy.loginfo(f"机器人底盘位姿: x={x:.2f}, y={y:.2f}, theta={theta/3.1415*180:.2f} degree")
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("TF lookup failed")


            # 2. 图像处理和语义分割
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            img_ori = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            H_ori, W_ori = img_ori.shape[:2]  # 720, 1280
            rospy.loginfo(f"原始图像尺寸: ({H_ori}, {W_ori})")
            # 关键：16:9比例缩放（640x360），保证上采样后无变形
            resize_size = (640, 360)
            img_resized = cv2.resize(img_ori, resize_size, interpolation=cv2.INTER_LINEAR)
            img_data = self.preprocess(img_resized)
            with torch.no_grad():
                feed_dict = {'img_data': img_data.unsqueeze(0).cuda(self.gpu)}
                pred = self.segmentation_module(feed_dict, segSize=(H_ori, W_ori))
                _, pred = torch.max(pred, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())
            # 生成sseg（720x1280，和原始图像/深度图对齐）
            # 优化后代码（极快）：
            pred_shifted = pred + 1  # 先做ID偏移（和原逻辑一致）
            # 过滤超出映射数组范围的ID（避免索引越界）
            pred_shifted_clipped = np.clip(pred_shifted, 0, len(self.id_mapping_array)-1)
            # 数组索引映射（向量化操作，无循环）
            sseg = self.id_mapping_array[pred_shifted_clipped]
            sseg = sseg.astype(np.int32)
            valid_mask = sseg != DEFAULT_NEW_IDX

            # 3. 深度点云投影
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            h, w = depth_img.shape
            # sample_val = depth_img[h//2, w//2]  # 中心像素值
            # rospy.loginfo(f"深度图中心像素值: {sample_val:.3f} 米 (编码: {depth_msg.encoding})")
            # points_cam: (N, 3), u, v: (N,)
            points_cam, u, v = self.pub3d.depth_to_pointcloud_camera(depth_img)
            # print(f"点云范围: x[{points_cam[:,0].min():.3f}, {points_cam[:,0].max():.3f}], "
            #             f"y[{points_cam[:,1].min():.3f}, {points_cam[:,1].max():.3f}], "
            #             f"z[{points_cam[:,2].min():.3f}, {points_cam[:,2].max():.3f}]") 
            # 将点云转换到 map 坐标系
            trans_world_cam = torch.tensor([
                transform.transform.translation.x,
                transform.transform.translation.y,
                1.12
            ], dtype=torch.float32, device=self.device)
            # 旋转矩阵GPU化
            quat = transform.transform.rotation
            R_world_footprint = torch.tensor(
                tf.transformations.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])[:3, :3],
                dtype=torch.float32, device=self.device
            )
            R_world_cam = R_world_footprint @ self.R_footprint_cam
            # 点云转GPU+批量矩阵乘法（核心加速）
            points_cam_tensor = torch.tensor(points_cam, dtype=torch.float32, device=self.device)
            # 旋转：(N,3) @ (3,3).T → (N,3)（批量运算，无循环）
            points_world_rot = points_cam_tensor @ R_world_cam.T
            # 平移：广播相加
            points_world_tensor = points_world_rot + trans_world_cam
            # 转回CPU（仅最后一步）
            points_world = points_world_tensor.cpu().numpy()
            flag_rviz_3dpoint = False
            if flag_rviz_3dpoint:
                # 发布点云到 RViz， 用于检查
                default_marker_id = 1
                if points_world.shape[0] > 10:
                    # pc_msg = create_colored_pointcloud_ZHJD(points_world, color=(255, 0, 0), frame_id="map")
                    # pc_pub.publish(pc_msg)
                    self.pub3d.publish_marker_pointcloud(self.marker_pub, points_world, marker_id=default_marker_id, color=(1, 1, 0), scale=0.01)


            # 4. 语义点云投影
            local3D = points_world[np.newaxis, ...]  # shape: [1, N, 3]
            local3D = torch.from_numpy(local3D).float().to("cuda")

            points2D = np.stack((u, v), axis=-1)[np.newaxis, ...]  # shape: [1, N, 2]
            points2D = torch.from_numpy(points2D).float().to("cuda")

            # 有效点云对应的语义分割像素点
            ssegs_3 = sseg[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
            ssegs_3 = torch.from_numpy(ssegs_3).float().to("cuda")

            # 六、地面投影，构建单帧语义栅格地图
            ego_semantic_sseg_27 = map_utils.ground_projection_ros(
                points2D, local3D, ssegs_3,
                sseg_labels=self.object_labels,
                grid_dim=self.grid_dim,
                cell_size=self.cell_size
            )  # shape: [t, 27, 184, 184]
            # print("ego_semantic_sseg_27.shape: ", ego_semantic_sseg_27.shape)  ego_semantic_sseg_27.shape:  torch.Size([1, 27, 200, 200])
            # self.semantic_map_publisher.publish_semantic_map(ego_semantic_sseg_27, res=0.1, origin_x=-10.0, origin_y=-10.0, height=-0.5)

            # # 七、累加到全局地图
            geo_semantic_sseg = ego_semantic_sseg_27
            step_geo_grid_sseg = self.sg.update_semantic_proj_grid_bayes(geo_grid=geo_semantic_sseg.unsqueeze(0))
            # print("step_geo_grid_sseg.shape: ", step_geo_grid_sseg.shape)  step_geo_grid_sseg.shape:  torch.Size([1, 1, 27, 200, 200])

            flag_rviz_2dmap = False
            if flag_rviz_2dmap:
                self.semantic_map_publisher.async_publish_semantic_map(
                    step_geo_grid_sseg.squeeze(0),  # 去掉批次维度，变成 [1, 27, 200, 200]
                    res=0.1, 
                    origin_x=-10.0, 
                    origin_y=-10.0, 
                    height=-0.5
                )




            # --- 帧率计算逻辑 ---
            curr_time = time.time()
            dt = curr_time - self.prev_time

            self.fps = 1.0 / dt if dt > 0 else 0.0
            self.prev_time = curr_time
            print(f"FPS: 【{self.fps:.2f}】 | Unique predictions: {np.unique(pred)} \n\n")

            # # 这里执行可视化逻辑...
            
            # pred_color = colorEncode(pred_new, colors_27).astype(np.uint8)
            # im_vis = np.concatenate((img_ori, pred_color), axis=1)
            # # cv2.imshow("ROS Segmentation", cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR))
            # # cv2.waitKey(1)
            # # 保存结果
            # save_path = os.path.join(self.cfg.TEST.result, f"{rgb_msg.header.seq}.png")
            # PILImage.fromarray(im_vis).save(save_path)

        except Exception as e:
            rospy.logerr(f"Inference error: {e}")
        
        finally:
            # --- 无论成功还是失败，锁都会在这里被释放 ---
            self.is_busy = False 
            # 如果你依然怀疑内存占用，可以在这里加上清理：
            # torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    rospy.init_node('ros_segmentation_node', anonymous=True)
    node = ROSSegmentationNode(cfg, args.gpu)
    
    image_sub = message_filters.Subscriber("/rgb/image_raw", ROSImage)
    depth_sub = message_filters.Subscriber("/depth_to_rgb/image_raw", ROSImage)
    # 同步器: slop=0.1 表示允许 100ms 的时间差
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=10, slop=0.1)
    ts.registerCallback(node.callback)

    
    rospy.loginfo(f"Node started, listening on {args.topic}")
    rospy.spin()