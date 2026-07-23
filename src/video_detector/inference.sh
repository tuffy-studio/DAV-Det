#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DATA_EVAL=""
OUTPUT_PATH="./video_prob.csv"

PRETRAIN_PATH="./weights/video_model.20.pth"
NUM_FRAMES=16
VIDEO_BATCH_SIZE=4
FRAME_BATCH_SIZE=64
IMG_SIZE=640
NUM_WORKERS=2


python inference.py \
  --data_eval ${DATA_EVAL} \
  --output_path ${OUTPUT_PATH} \
  --pretrain_path ${PRETRAIN_PATH} \
  --num_frames ${NUM_FRAMES} \
  --frame_batch_size ${FRAME_BATCH_SIZE} \
  --video_batch_size ${VIDEO_BATCH_SIZE} \
  --img_size ${IMG_SIZE} \
  --mode ${MODE}
