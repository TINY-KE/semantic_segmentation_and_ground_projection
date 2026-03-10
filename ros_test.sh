#!/bin/bash

# 模型配置
MODEL_NAME=ade20k-resnet50dilated-ppm_deepsup
MODEL_PATH=ckpt/$MODEL_NAME
RESULT_PATH=./

# 模型权重文件
ENCODER="${MODEL_PATH}/encoder_epoch_20.pth"
DECODER="${MODEL_PATH}/decoder_epoch_20.pth"

# ROS 图像话题（默认为常见话题）
ROS_IMAGE_TOPIC=${ROS_IMAGE_TOPIC:-"/rgb/image_raw"}

# 创建模型目录
if [ ! -e $MODEL_PATH ]; then
  mkdir -p $MODEL_PATH
  echo "创建模型目录: $MODEL_PATH"
fi

# 下载模型权重（如果不存在）
if [ ! -e $ENCODER ]; then
  echo "下载编码器权重..."
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
  if [ $? -ne 0 ]; then
    echo "警告: 无法从默认URL下载编码器权重"
  fi
fi

if [ ! -e $DECODER ]; then
  echo "下载解码器权重..."
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
  if [ $? -ne 0 ]; then
    echo "警告: 无法从默认URL下载解码器权重"
  fi
fi

# 检查模型权重是否存在
if [ ! -e $ENCODER ]; then
  echo "错误: 编码器权重文件不存在: $ENCODER"
  echo "请手动下载权重文件并放置到正确位置"
  exit 1
fi

if [ ! -e $DECODER ]; then
  echo "错误: 解码器权重文件不存在: $DECODER"
  echo "请手动下载权重文件并放置到正确位置"
  exit 1
fi

echo "模型权重已就绪"
echo "编码器: $ENCODER"
echo "解码器: $DECODER"

# 设置结果保存目录
RESULT_DIR="./save_results/ros_segmentation_online"
if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
  echo "创建结果保存目录: $RESULT_DIR"
fi

# 检查ROS环境
if [ -z "$ROS_DISTRO" ]; then
  echo "警告: ROS环境未设置，请先运行 'source /opt/ros/<distro>/setup.bash'"
  echo "将尝试自动检测ROS环境..."
  
  # 尝试自动检测ROS版本
  if [ -d "/opt/ros/noetic" ]; then
    source /opt/ros/noetic/setup.bash
    echo "检测到ROS Noetic，已自动设置环境"
  elif [ -d "/opt/ros/melodic" ]; then
    source /opt/ros/melodic/setup.bash
    echo "检测到ROS Melodic，已自动设置环境"
  elif [ -d "/opt/ros/kinetic" ]; then
    source /opt/ros/kinetic/setup.bash
    echo "检测到ROS Kinetic，已自动设置环境"
  elif [ -d "/opt/ros/foxy" ]; then
    source /opt/ros/foxy/setup.bash
    echo "检测到ROS2 Foxy，已自动设置环境"
  else
    echo "错误: 未找到ROS安装，请确保ROS已正确安装和配置"
    exit 1
  fi
fi

echo "ROS环境已设置: $ROS_DISTRO"
echo "订阅图像话题: $ROS_IMAGE_TOPIC"
echo "结果保存到: $RESULT_DIR"


# 运行语义分割节点
python3 -u ground_projection_online/ros_segment_and_project3d.py \
  --topic "$ROS_IMAGE_TOPIC" \
  --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml \
  --gpu 0 \
  DIR $MODEL_PATH \
  TEST.result $RESULT_DIR \
  TEST.checkpoint epoch_20.pth \
  $@