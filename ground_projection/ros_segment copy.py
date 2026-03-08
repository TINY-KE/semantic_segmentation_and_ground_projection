# System libs
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg
from ground_projection.util import viz_utils, map_utils, utils, Id_Converter


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

class ROSSegmentationModule:
    def __init__(self, cfg, gpu):
        self.cfg = cfg
        self.gpu = gpu
        self.bridge = CvBridge()
        
        # 构建模型 (保留你原有的构建逻辑)
        self.net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder, fc_dim=cfg.MODEL.fc_dim, weights=cfg.MODEL.weights_encoder)
        self.net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder, fc_dim=cfg.MODEL.fc_dim, num_class=cfg.DATASET.num_class, 
            weights=cfg.MODEL.weights_decoder, use_softmax=True)
        self.segmentation_module = SegmentationModule(self.net_encoder, self.net_decoder, None)
        self.segmentation_module.cuda(self.gpu)
        self.segmentation_module.eval()

        # 订阅 ROS 话题
        self.sub = rospy.Subscriber("/rgb/image_raw", Image, self.callback, queue_size=1)

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_ori = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 预处理 (与你原程序逻辑一致)
            img_data = self.preprocess(img_ori)
            segSize = (img_ori.shape[0], img_ori.shape[1])
            
            with torch.no_grad():
                feed_dict = {'img_data': img_data.unsqueeze(0).cuda(self.gpu)}
                pred = self.segmentation_module(feed_dict, segSize=segSize)
                _, pred = torch.max(pred, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())
            
            # 使用你原有的可视化函数
            self.visualize_result(img_ori, pred)
            
        except Exception as e:
            rospy.logerr(f"Callback error: {e}")

    def preprocess(self, img):
        # 请根据你原程序里的 Dataset 逻辑适配，这里给出通用归一化
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(img.transpose(2, 0, 1))

    def visualize_result(self, img, pred):
        # 直接保留你原有的 visualize_result 逻辑
        pred_new = np.vectorize(lambda x: old_to_new_idx.get(x + 1, DEFAULT_NEW_IDX))(pred)
        pixs = pred_new.size
        uniques, counts = np.unique(pred_new, return_counts=True)
        pred_color = colorEncode(pred_new, colors_27).astype(np.uint8)
        
        # 显示结果
        im_vis = np.concatenate((img, pred_color), axis=1)
        cv2.imshow("Result", cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

if __name__ == '__main__':
    # 1. 命令行参数解析 (完全保留)
    parser = argparse.ArgumentParser(description="ROS Semantic Segmentation")
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.weights_encoder = f"{cfg.DIR}/encoder_{cfg.TEST.checkpoint}"
    cfg.MODEL.weights_decoder = f"{cfg.DIR}/decoder_{cfg.TEST.checkpoint}"

    # 2. 初始化 ROS 节点
    rospy.init_node('segmentation_node', anonymous=True)
    node = ROSSegmentationModule(cfg, args.gpu)
    
    rospy.loginfo("Segmentation node running...")
    rospy.spin()