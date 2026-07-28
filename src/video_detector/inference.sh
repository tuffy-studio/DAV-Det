#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DATA_EVAL="" # TODO
OUTPUT_PATH=""  # TODO
PRETRAIN_PATH=""  # TODO，训练完成模型的权重路径
NUM_FRAMES=16
VIDEO_BATCH_SIZE=4
FRAME_BATCH_SIZE=64
IMG_SIZE=640
NUM_WORKERS=2
backbone_configure="./dinov3-vitl16-pretrain-lvd1689m"

python inference.py \
  --data_eval ${DATA_EVAL} \
  --output_path ${OUTPUT_PATH} \
  --pretrain_path ${PRETRAIN_PATH} \
  --num_frames ${NUM_FRAMES} \
  --frame_batch_size ${FRAME_BATCH_SIZE} \
  --video_batch_size ${VIDEO_BATCH_SIZE} \
  --img_size ${IMG_SIZE} \
  --backbone_configure ${backbone_configure}
