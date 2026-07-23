#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
# ==================== Paths ====================

# 输入 CSV：第一列为 file_path
data_eval="" # TODO: set the path to your evaluation data

# 输出概率 CSV
output_path="./output_image_prob.csv"

# GPS_DINO 微调权重路径
pretrain_path="" # TODO: set the path to the fine-tuned weights

# DINOv3 预训练权重目录
backbone_configure="./dinov3-vitl16-pretrain-lvd1689m"

# ==================== Inference Params ====================

batch_size=256
num_workers=16
img_size=224 # 224 for Fakeavceleb, 640 for MVAD
write_mode="stream"   # stream: 逐 batch 写入；batch: 最后一次性写入

# 是否使用水平翻转 TTA
# TTA 时会对原图和翻转图的 logits 取平均后再 sigmoid
use_tta=False

# ==================== Run ====================

cmd="python ./inference_image.py \
  --data_eval ${data_eval} \
  --output_path ${output_path} \
  --pretrain_path ${pretrain_path} \
  --backbone_configure ${backbone_configure} \
  --batch_size ${batch_size} \
  --num_workers ${num_workers} \
  --img_size ${img_size} \
  --write_mode ${write_mode}"

if [ "$use_tta" = "True" ]; then
    cmd="${cmd} --if_test_time_augment"
fi

echo "Running: ${cmd}"
eval ${cmd}
