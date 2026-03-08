#!/bin/bash

# Image and model names
#TEST_IMG=ADE_val_00001519.jpg
#TEST_IMG=test.jpg
#TEST_IMG=/home/robotlab/work/semantic-segmentation-pytorch/RGB_images/rgb
TEST_IMG=/home/robotlab/work/semantic-segmentation-pytorch/RGB_images/icl
#TEST_IMG=/home/robotlab/work/semantic-segmentation-pytorch/RGB_images/NPZ_test
#TEST_IMG=/home/robotlab/work/semantic-segmentation-pytorch/RGB_images/binzhou_wjl/
#TEST_IMG=/home/robotlab/work/semantic-segmentation-pytorch/RGB_images/toilet/


#ade20k-hrnetv2.yaml                        ade20k-resnet18dilated-ppm_deepsup.yaml
#ade20k-mobilenetv2dilated-c1_deepsup.yaml  ade20k-resnet50dilated-ppm_deepsup.yaml
#ade20k-resnet101dilated-ppm_deepsup.yaml   ade20k-resnet50-upernet.yaml
#ade20k-resnet101-upernet.yaml
# 编码器名称	说明
  #MobileNetV2dilated	轻量级网络，适合边缘设备。使用膨胀卷积扩展感受野。
  #ResNet18 / ResNet18dilated	浅层残差网络，dilated 版本用于保留高分辨率。
  #ResNet50 / ResNet50dilated	常用中层网络，dilated 版本适合分割任务，感受野更大。
  #ResNet101 / ResNet101dilated	更深的网络，具有更强的特征提取能力。
  #HRNetV2 (W48)	高分辨率网络，能持续保持高分辨率特征，适合精细分割。
MODEL_NAME=ade20k-resnet50dilated-ppm_deepsup
#MODEL_NAME=ade20k-resnet101dilated-ppm_deepsup
#http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-mobilenetv2dilated-c1_deepsup/encoder_epoch_20.pth
#http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet101dilated-ppm_deepsup/encoder_epoch_20.pth
#http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2/encoder_epoch_20.pth

MODEL_PATH=ckpt/$MODEL_NAME
RESULT_PATH=./

#ENCODER=$MODEL_NAME/encoder_epoch_20.pth
#DECODER=$MODEL_NAME/decoder_epoch_20.pth
# 只需修改这两个变量
ENCODER="${MODEL_PATH}/encoder_epoch_20.pth"  # 改为绝对路径
DECODER="${MODEL_PATH}/decoder_epoch_20.pth"  # 改为绝对路径

# Download model weights and image
if [ ! -e $MODEL_PATH ]; then
  mkdir -p $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi
if [ ! -e $TEST_IMG ]; then
  wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016/images/validation/$TEST_IMG
fi

if [ -z "$DOWNLOAD_ONLY" ]
then

## Inference
#python3 -u test.py \
#  --imgs $TEST_IMG \
#  --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml \
#  DIR $MODEL_PATH \
#  TEST.result ./save_results/temp \
#  TEST.checkpoint epoch_20.pth

python3 -u ground_projection/mit_segment_NPZ.py \
  --imgs $TEST_IMG \
  --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml \
  DIR $MODEL_PATH \
  TEST.result ./save_results/temp \
  TEST.checkpoint epoch_20.pth

fi
