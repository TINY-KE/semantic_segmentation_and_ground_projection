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
from ground_projection.util import viz_utils, map_utils, utils

# old_idx → new_idx 映射表（只包含有效项）
DEFAULT_NEW_IDX = 0  # 默认类别（void）
# old_to_new_idx = {
#     20: 1,    # 椅子
#     31: 1,
#     76: 1,
#     111: 1,
#     15: 2,    # 门
#     34: 3,    # 桌子
#     16: 3,
#     65: 3,
#     40: 4,    # 靠垫cushion
#     58: 4,
#     24: 5,    # 沙发
#     8: 6,     # 床
#     # 18: 7,    # 植物
#     # 48: 8,    # 水槽
#     90: 10,   # 电视
#     # 146: 11,  # shower 淋浴
#     38: 12,   # 浴缸 bathtub
#     46: 13,   # 柜台 counter
#     71: 13,
#     100: 13,
#     1: 15,    # 墙 structure
#     # 4: 17,    # 地板 free-space
#     # 14: 17,
#     # 23: 18,   # 画
#     11: 19,   # 橱柜 cabinet
#     # 50: 23,   # 壁炉 fireplace
#     # 82: 22    # 毛巾 towel
# }

# 注意old idx需要减一
old_to_new_idx = {
    20: 1,    # 椅子
    31: 1,
    76: 1,
    111: 1,
    15: 2,    # 门
    34: 3,    # 桌子
    16: 3,
    65: 3,
    40: 4,    # 靠垫cushion
    58: 4,
    24: 5,    # 沙发
    8: 6,     # 床
    18: 7,    # 植物
    48: 8,    # 水槽
    90: 10,   # 电视
    146: 11,  # shower 淋浴
    38: 12,   # 浴缸 bathtub
    46: 13,   # 柜台 counter
    71: 13,
    100: 13,
    1: 15,    # 墙 structure
    4: 17,    # 地板 free-space
    14: 17,
    29: 17,
    95: 17,
    23: 18,   # 画
    11: 19,   # 橱柜 cabinet
    50: 23,   # 壁炉 fireplace
    82: 22    # 毛巾 towel
}

color_mapping_27 = {
    0:  (255, 255, 255),   # 白色 white                       空类别 / 无类别 (void)
    1:  (128, 128, 0),     # 橄榄色 olive                     椅子 (chair)
    2:  (0, 0, 255),       # 蓝色 blue                        门 (door)
    3:  (255, 0, 0),       # 红色 red                         桌子 (table)
    4:  (255, 0, 255),     # 洋红色 magenta                   靠垫 / 坐垫 (cushion)
    5:  (0, 255, 255),     # 青色 cyan                        沙发 (sofa)
    6:  (255, 165, 0),     # 橙色 orange                      床 (bed)
    7:  (255, 255, 0),     # 黄色 yellow                      植物 (plant)
    8:  (128, 128, 128),   # 灰色 gray                        洗手池 / 水槽 (sink)
    9:  (128, 0, 0),       # 栗色 maroon                      马桶 (toilet)
    10: (255, 20, 147),    # 深粉红 deep pink                 电视 / 显示器 (tv_monitor)
    11: (0, 128, 0),       # 深绿色 dark green               淋浴器 (shower)
    12: (128, 0, 128),     # 紫色 purple                      浴缸 (bathtub)
    13: (0, 128, 128),     # 水鸭色 teal                      操作台 / 工作台 (counter)
    14: (0, 0, 128),       # 藏青色 navy                     家电 (appliances)
    15: (210, 105, 30),    # 巧克力色 chocolate              建筑结构 (structure)
    16: (188, 143, 143),   # 褐玫瑰色 rosy brown             其他 / 杂项 (other)
    17: (0, 255, 0),       # 绿色 green                      空闲空间 / 可行走区域 (free-space)   ****
    18: (255, 215, 0),     # 金色 gold                       图片 / 挂画 (picture)
    19: (0, 0, 0),         # 黑色 black                      橱柜 / 柜子 (cabinet)
    20: (192, 192, 192),   # 银色 silver                     抽屉柜 (chest_of_drawers)
    21: (138, 43, 226),    # 蓝紫色 blue violet              凳子 (stool)
    22: (255, 127, 80),    # 珊瑚色 coral                    毛巾 (towel)
    23: (238, 130, 238),   # 紫罗兰色 violet                 壁炉 (fireplace)
    24: (245, 245, 220),   # 米色 / 浅卡其 beige            健身器材 (gym_equipment)
    25: (139, 69, 19),     # 马鞍棕 saddle brown            座位（综合类）(seating)
    26: (64, 224, 208)     # 绿松石色 turquoise              衣物 (clothes)
}

# 转换为 numpy 数组（按 key 排序，确保顺序一致）
colors_27 = np.array([color_mapping_27[i] for i in sorted(color_mapping_27.keys())], dtype=np.uint8)

def extract_number(fpath):
    """提取文件名中数字部分作为排序依据"""
    base = os.path.basename(fpath)           # e.g. "126.png"
    name, _ = os.path.splitext(base)         # "126"
    return int(name)                         # 转成整数进行排序

def get_sorted_numeric_filenames(folder_path):
    filenames = os.listdir(folder_path)

    # 只保留文件（排除子目录），且文件名是数字加扩展名（如 1.png, 2.jpg）
    numeric_files = []
    for f in filenames:
        full_path = os.path.join(folder_path, f)
        if os.path.isfile(full_path):
            name, ext = os.path.splitext(f)
            if name.isdigit():  # 文件名是纯数字
                numeric_files.append(f)

    # 按文件名前的数字部分排序，例如 "10.jpg" > "2.jpg"
    numeric_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    return numeric_files

def visualize_result(data, pred, cfg):
    (img, info) = data

    pred = np.int32(pred)
    print("     [zhjd-debug] pred.shape：", pred.shape)   #  [zhjd-debug] pred.shape： (540, 960)
    # ✨ 新增：将 old idx 转成 new idx
    pred_new = np.vectorize(lambda x: old_to_new_idx.get(x + 1, DEFAULT_NEW_IDX))(pred)
    # 注意：原始预测类别是从 0 开始，而 old idx 是从 1 开始，所以 pred + 1

    # 获取预测中所有出现的类别及其像素数量
    pixs = pred_new.size
    uniques, counts = np.unique(pred_new, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        new_id = uniques[idx]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  new_id {}: {:.2f}%".format(new_id, ratio))

    # 上色（你可以根据 new_id 自定义颜色映射）
    pred_color = colorEncode(pred_new, colors_27).astype(np.uint8)

    # 拼接原图和预测图
    im_vis = np.concatenate((img, pred_color), axis=1)
    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))

#segmentation_module：语义分割模型（包含 encoder + decoder）
# loader：测试数据的 DataLoader
# gpu：使用的 GPU 设备编号
def inference(segmentation_module, loader, gpu):
    all_imgs = []
    all_ssegs = []

    # 设置模型为评估状态（eval()），关闭 dropout、batchnorm 的更新等行为。
    segmentation_module.eval()

    #  tqdm 进度条
    pbar = tqdm(total=len(loader))

    # 遍历测试数据集
    for batch_data in loader:
        # process data
        # 每次从 DataLoader 中取出一个 batch（实际上这里是每次只处理一张图）
        # batch_data[0] 是真正的图像数据（因为使用了 user_scattered_collate）
        batch_data = batch_data[0]
        # img_ori 是原始图像（numpy 数组）
        # segSize 是原图的尺寸 (H, W)，用于模型输出 resize
        # [zhjd-debug] 说明在输入进segmentation_module之前，先把图像变成正方形。
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        # print('     [zhjd-debug] segSize：', segSize)
        img_resized_list = batch_data['img_data']
        # print("     [zhjd-debug] img_resized_list 类型：", type(img_resized_list))
        # 多尺度推理的好处
        # 模型在不同输入尺寸下可能有不同的预测结果
        # 多尺度预测后求平均（或投票），可以提高最终分割的稳定性和准确性
        # 是一种常见的测试时增强（Test-Time Augmentation, TTA）
        # print("包含的图像数量：", len(img_resized_list))
        # for i, img in enumerate(img_resized_list):
        #     print(f"第 {i} 个图像类型：{type(img)}, 尺寸：{img.shape}")

        with torch.no_grad():
            # 初始化空分数图（所有类）
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                # 构建 feed_dict，只包含模型需要的键（去除原始图等无关项）
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                # 将数据移到 GPU 上
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                # 模型前向推理，得到每类的分数图 pred_tmp
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                print("     [zhjd-debug] pred_tmp 维度：", pred_tmp.shape)  # [zhjd-debug] pred_tmp 维度： torch.Size([1, 150, 574, 860])
                #  将多尺度推理的结果进行平均融合
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)   # len(cfg.DATASET.imgSizes)为尺度数量，等于5。

            # 获取最终预测类别图
            _, pred = torch.max(scores, dim=1)   # 从 scores 的类别维度上取最大值索引 → 得到预测类别 ID
            # squeeze(0) 移除 batch 维度
            # as_numpy() 转成 NumPy 格式，方便后续处理
            pred = as_numpy(pred.squeeze(0).cpu())

            # 保存原图和语义图到 .npz 文件
            img_ori = batch_data['img_ori']  # 原始图像
            all_imgs.append(img_ori)
            pred_2 = np.int32(pred)
            pred_3 = np.vectorize(lambda x: old_to_new_idx.get(x + 1, DEFAULT_NEW_IDX))(pred_2)  # ✨ 新增：将 old idx 转成 new idx
            all_ssegs.append(pred_3)

        # 保存为本地图片
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )

        pbar.update(1)  #更新进度条

    # 保存所有图像到一个 NPZ 文件
    #     # work2: ['abs_pose', 'ego_grid_crops_spatial', 'step_ego_grid_crops_spatial', 'gt_grid_crops_spatial', 'gt_grid_crops_objects',
    #     'images', 'ssegs', 'depth_imgs', 'pred_ego_crops_sseg', 'step_ego_grid_27']
    save_path = os.path.join(cfg.TEST.result, "all_data.npz")
    np.savez_compressed(save_path,
                        imgs=np.stack(all_imgs),  # (N, H, W, 3)
                        ssegs=np.stack(all_ssegs))  # (N, H, W)
    print(f"\n✅ Saved all data to: {save_path}")


def SegmentationModuleNet(cfg, gpu):
    # 使用配置 cfg 和指定的 GPU，构建模型并在测试集上运行推理，输出语义分割预测结果。
    torch.cuda.set_device(gpu)

    # 2. Network Builders
    # 编码器（Encoder）
    # 构建主干网络（如 dilated ResNet-50）。
    # 加载预训练权重。
    # fc_dim 是输出特征维度（如 2048）。
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    # 解码器（Decoder）
    # 构建解码器（如 PPM + Deep Supervision）。
    # 使用 softmax 输出最终类别概率。
    # 支持多类分割（num_class）。
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    # 3. 损失函数（虽然用于测试中可能不使用）
    crit = nn.NLLLoss(ignore_index=-1)
    #  4. 封装为 SegmentationModule
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    # 5. 构建测试数据集
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)

    #  6. 构建 DataLoader
    print("     [zhjd-debug] batch_size：",cfg.TEST.batch_size)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,  #  batch_size = 1
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)   # drop_last=True：丢弃最后一个不完整的 batch（如果有）。


    # 7. 模型放入 GPU
    segmentation_module.cuda()

    # Main loop
    #  8. 执行推理
    inference(segmentation_module, loader_test, gpu)

    print('Inference done!')


import os


def load_pairs_from_association_file(file_path, imgs_root_path, valid_timestamps):
    """
    从关联文件中加载符合条件的 RGB 和深度图路径对。

    参数:
        file_path (str): 关联文件路径
        imgs_root_path (str): 图像根目录
        valid_timestamps (set or list): 有效的时间戳集合

    返回:
        list_depth: [{'timestamp': ..., 'fpath_img': ...}, ...]
        list_rgb:   [{'timestamp': ..., 'fpath_img': ...}, ...]
    """
    print(f"Loading pairs from association file: {file_path}")

    list_depth = []
    list_rgb = []

    # 确保 valid_timestamps 是 set，提高查找效率
    if not isinstance(valid_timestamps, set):
        valid_timestamps = set(valid_timestamps)

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 4:
                print(f"[Line {line_num}] 跳过格式错误的行: {line.strip()}")
                continue

            timestamp, depth_path, _, rgb_path = parts[:4]
            timestamp = int(timestamp)
            if timestamp in valid_timestamps:
                # 构造完整路径
                depth_full_path = os.path.join(imgs_root_path, depth_path)
                rgb_full_path = os.path.join(imgs_root_path, rgb_path)

                list_depth.append({
                    'timestamp': timestamp,
                    'fpath_img': depth_full_path
                })
                list_rgb.append({
                    'timestamp': timestamp,
                    'fpath_img': rgb_full_path
                })
            else:
                print(f"[Warning] timestamp {timestamp} 不在 valid_timestamps 中")
                print(f"valid_timestamps 中的例子: {list(valid_timestamps)[:5]}")  # 只显示前5个

    print(f"✅ 成功加载 {len(list_rgb)} 对图像")
    return list_depth, list_rgb

def load_poses_from_file(poses_file):
    """
    从 KeyFrames_for_smp.txt 文件中读取位姿数据。

    每行格式: time x y z qx qy qz qw
    返回:
        poses: List[Tuple[int, List[float]]]，每个元素是 (time, [x, y, z, qx, qy, qz, qw])
    """
    poses = []
    valid_timestamps = []

    try:
        with open(poses_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # 跳过空行或注释
                parts = line.split()
                if len(parts) != 8:
                    print(f"[warning] 跳过格式错误的行: {line}")
                    continue
                try:
                    time = int(parts[0])
                    pose = list(map(float, parts[1:]))
                    translation = pose[:3]  # 位移: [x, y, z]
                    quaternion = pose[3:]  # 四元数: [qx, qy, qz, qw]
                    valid_timestamps.append(time)
                    poses.append((translation, quaternion))
                except ValueError:
                    print(f"[error] 无法解析该行: {line}")
    except FileNotFoundError:
        print(f"[error] 文件未找到: {poses_file}")

    return valid_timestamps, poses

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    # --imgs	必须参数，图片路径或文件夹路径
    # --cfg	模型配置文件路径，默认为 config/ade20k-resnet50dilated-ppm_deepsup.yaml
    # --gpu	使用的 GPU ID，默认为 0
    # opts	可以通过命令行动态修改配置项，例如 MODEL.fc_dim 512
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs_root",
        required=True,
        type=str,
        help="an image path, or a directory name"
    )
    parser.add_argument(
        "--association",
        default="associations.txt",
        metavar="FILE",
        help="path to associations.txt",
        type=str,
    )
    parser.add_argument(
        "--pose_groundtruth",
        default="associations.txt",
        metavar="FILE",
        help="path to pose_groundtruth.txt",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(list(cfg.keys()))

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))


    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # 配置模型权重路径
    # 根据 cfg.DIR 和 checkpoint 名称拼接出模型权重文件的绝对路径。
    # 同时通过 assert 检查文件是否存在。
    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # RGBD的根目录
    if not os.path.isdir(args.imgs_root):
        print(f"❌ 图片地址不存在")

    # 获取位置信息
    poses_file = args.imgs_root + '/' + 'KeyFrames_for_smp.txt'
    cfg.valid_timestamps, cfg.poses  = load_poses_from_file(poses_file)
    # for time, (translation, quaternion) in zip(cfg.valid_timestamps, cfg.poses):
    #     print(f"VALID  time={time}, translation={translation}, quaternion={quaternion}")

    # 从association文件中读取图片
    associations_file = args.imgs_root + '/' + args.association
    cfg.list_depth, cfg.list_rgb = load_pairs_from_association_file(associations_file, args.imgs_root, cfg.valid_timestamps)
    # print("cfg.valid_timestamps: ", cfg.valid_timestamps)
    # print("[zhjd-debug] cfg.list_depth:", cfg.list_depth)
    # print("[zhjd-debug] cfg.list_rgb:  ", cfg.list_rgb)
    # for time, (translation, quaternion), image in zip(cfg.valid_timestamps, cfg.poses, cfg.list_rgb):
    #     print(f"VALID  time={time}, translation={translation}, quaternion={quaternion}, image={image}")

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    # 调用main函数，开始推理
    # SegmentationModuleNet(cfg, args.gpu)

    print("     [zhjd-debug] 结果的保存地址: ", cfg.TEST.result)
