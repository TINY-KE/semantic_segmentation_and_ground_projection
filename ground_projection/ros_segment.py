#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ROSImage
import cv2

# System libs
import sys
import os
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time # 在文件开头添加 import time
import argparse
# Numerical libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from mit_semseg.lib.utils import as_numpy
from PIL import Image as PILImage # 别名，解决命名冲突
from mit_semseg.config import cfg
from ground_projection.util import Id_Converter

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
        self.device = torch.device(f"cuda:{gpu}")
        self.is_busy = False  # --- 新增：忙碌状态锁 ---
        self.prev_time = time.time() # 初始化时间戳
        self.fps = 0.0

        # 构建模型
        self.net_encoder = ModelBuilder.build_encoder(arch=cfg.MODEL.arch_encoder, fc_dim=cfg.MODEL.fc_dim, weights=cfg.MODEL.weights_encoder)
        self.net_decoder = ModelBuilder.build_decoder(arch=cfg.MODEL.arch_decoder, fc_dim=cfg.MODEL.fc_dim, num_class=cfg.DATASET.num_class, weights=cfg.MODEL.weights_decoder, use_softmax=True)
        self.segmentation_module = SegmentationModule(self.net_encoder, self.net_decoder, None)
        self.segmentation_module.cuda(self.gpu)
        self.segmentation_module.eval()

    def preprocess(self, img):
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1))
    
    def callback(self, msg):
        # --- 新增：只处理最新图像 ---
        if self.is_busy:
            return  # 如果上一帧还没处理完，直接跳过当前这一帧
        
        self.is_busy = True
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_ori = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_ori, (512, 512)) 
            img_data = self.preprocess(img_resized) # 用 resize 后的图计算
            
            with torch.no_grad():
                feed_dict = {'img_data': img_data.unsqueeze(0).cuda(self.gpu)}
                pred = self.segmentation_module(feed_dict, segSize=(img_ori.shape[0], img_ori.shape[1]))
                _, pred = torch.max(pred, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())
            
            
            # # 保存结果
            # save_path = os.path.join(self.cfg.TEST.result, f"{msg.header.seq}.png")
            # PILImage.fromarray(im_vis).save(save_path)
                        

            # --- 帧率计算逻辑 ---
            curr_time = time.time()
            dt = curr_time - self.prev_time
            self.fps = 1.0 / dt if dt > 0 else 0.0
            self.prev_time = curr_time
            print(f"FPS: 【{self.fps:.2f}】 | Unique predictions: {np.unique(pred)}")

            # 这里执行可视化逻辑...
            # pred_new = np.vectorize(lambda x: old_to_new_idx.get(x + 1, DEFAULT_NEW_IDX))(pred)
            # pred_color = colorEncode(pred_new, colors_27).astype(np.uint8)
            # im_vis = np.concatenate((img_ori, pred_color), axis=1)
            # cv2.imshow("ROS Segmentation", cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)

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
    
    # 订阅使用 ROSImage 别名
    rospy.Subscriber(args.topic, ROSImage, node.callback, queue_size=1)
    
    rospy.loginfo(f"Node started, listening on {args.topic}")
    rospy.spin()