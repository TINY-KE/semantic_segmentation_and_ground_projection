# System libs
import os
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

# 3. 加载颜色映射和类别名称
# 加载颜色映射表（用于语义分割结果上色）；
# 读取类别信息文件（object150_info.csv），并提取每个类别的主名称；
# 构建一个 names 字典，用于根据类别 ID 查询主类名。
colors = loadmat('data/color150.mat')['colors']  #loadmat() 会将 .mat 文件转换为一个 Python 字典。
names = {}  #
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)   # 构建一个逐行读取器。
    next(reader)  # 跳过第一行标题行（Idx, Ratio, Train, ...）
    for row in reader:   #遍历每一行数据（类型是 list[str]），例如： row = ['2', '0.1072', '6046', '612', '1', 'building;edifice']
        names[int(row[0])] = row[5].split(";")[0]   # 取第一个名称（如 "building"）
    # 最终构建一个字典names：
    # names = {
    #     1: 'wall',
    #     2: 'building',
    #     3: 'sky',
    #     ...
    #     13: 'person'
    # }

#  4. 可视化分割结果
# 作用：
# 统计并打印预测图中每个类别所占比例。
# 把原图和分割上色图拼接并保存为 PNG 文件。
def visualize_result(data, pred, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    # 获取预测中所有出现的类别及其像素数量
    #     pixs：预测图中像素总数（即高 × 宽）
    #     uniques：预测中出现的类别标签值
    #     counts：每个类别对应的像素数量
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:  # 只打印占比超过 0.1% 的类别
            print("  {}: {:.2f}%".format(name, ratio))

    # 将预测结果“上色”
    # colorEncode(pred, colors)：把每个类别 ID 映射为 RGB 颜色（比如 0 是黑色，1 是红色等）
    # 返回的是一个彩色预测图 pred_color
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    # 拼接原图和预测图
    im_vis = np.concatenate((img, pred_color), axis=1)

    # 构造输出图像名称
    img_name = info.split('/')[-1]
    # 保存结果图像
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))


#segmentation_module：语义分割模型（包含 encoder + decoder）
# loader：测试数据的 DataLoader
# gpu：使用的 GPU 设备编号
def test(segmentation_module, loader, gpu):
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
        print('     [zhjd-debug] segSize：', segSize)
        img_resized_list = batch_data['img_data']
        print("     [zhjd-debug] img_resized_list 类型：", type(img_resized_list))
        # 多尺度推理的好处
        # 模型在不同输入尺寸下可能有不同的预测结果
        # 多尺度预测后求平均（或投票），可以提高最终分割的稳定性和准确性
        # 是一种常见的测试时增强（Test-Time Augmentation, TTA）
        print("包含的图像数量：", len(img_resized_list))
        for i, img in enumerate(img_resized_list):
            print(f"第 {i} 个图像类型：{type(img)}, 尺寸：{img.shape}")

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

        # visualization
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )

        pbar.update(1)  #更新进度条


def main(cfg, gpu):
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
    test(segmentation_module, loader_test, gpu)

    print('Inference done!')


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
        "--imgs",
        required=True,
        type=str,
        help="an image path, or a directory name"
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

    # 命令行中输入的要处理的图片
    # 如果是文件夹，就递归查找所有图片；
    # 否则就是一张图；
    # 最终构造成一个
    # list，每个元素是
    # {'fpath_img': 路径}。
    # generate testing image list
    if os.path.isdir(args.imgs):
        imgs = find_recursive(args.imgs)
    else:
        imgs = [args.imgs]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)
    print("     [zhjd-debug] cfg.TEST.result: ",cfg.TEST.result)
    # 调用main函数，开始推理
    main(cfg, args.gpu)
