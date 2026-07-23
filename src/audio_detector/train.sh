#!/bin/bash
set -e

# GPU selection: physical GPUs visible to CUDA; override with env var, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
SAVE_DIR="" # TODO
TRAIN_CSV="" # TODO
DEV_CSV="" # TODO
PEAV_CHECKPOINT="./pe-av-base"

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=512
NUM_WORKERS=16
LR_HEAD=1e-4
LR_LORA=1e-4
WEIGHT_DECAY=5e-2
GRAD_CLIP=1.0
WARMUP_RATIO=1.0
ACCUM_STEPS=1

# Model config
CLIP_LENGTH=144000
SAMPLING_RATE=48000

# LoRA config
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.1

# Audio detector backend config
NUM_HEADS=8
ATTN_DROPOUT=0.1

# Deep supervision
USE_DEEP_SUPERVISION=false
NUM_SUPERVISION_LAYERS=3 # only used if USE_DEEP_SUPERVISION is true
AUX_LOSS_WEIGHT=0.0

# Loss function
LOSS_TYPE=focal
FOCAL_ALPHA=0.6 # 0.6 for MVAD, 0.5 for Fakeavceleb
FOCAL_GAMMA_POS=2.0
FOCAL_GAMMA_NEG=2.0

# Data augmentation
AUGMENT=true
AUGMENT_INTENSITY=5
NUM_AUGMENT=3

# Device
DEVICE=cuda

# Create save directory
mkdir -p ${SAVE_DIR}

echo "=========================================="
echo "Training Audio Deepfake Detector (Full)"
echo "=========================================="
echo "Save dir: ${SAVE_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Deep supervision: ${USE_DEEP_SUPERVISION}"
echo "Loss: ${LOSS_TYPE}"
echo "=========================================="

python3 train.py \
    --train_csv ${TRAIN_CSV} \
    --dev_csv ${DEV_CSV} \
    --peav_checkpoint ${PEAV_CHECKPOINT} \
    --save_dir ${SAVE_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr_head ${LR_HEAD} \
    --lr_lora ${LR_LORA} \
    --weight_decay ${WEIGHT_DECAY} \
    --grad_clip ${GRAD_CLIP} \
    --warmup_ratio ${WARMUP_RATIO} \
    --accum_steps ${ACCUM_STEPS} \
    --clip_length ${CLIP_LENGTH} \
    --sampling_rate ${SAMPLING_RATE} \
    --use_lora \
    --freeze_extractor \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --unfreeze_norm \
    --num_heads ${NUM_HEADS} \
    --attn_dropout ${ATTN_DROPOUT} \
    --use_deep_supervision \
    --num_supervision_layers ${NUM_SUPERVISION_LAYERS} \
    --aux_loss_weight ${AUX_LOSS_WEIGHT} \
    --loss_type ${LOSS_TYPE} \
    --focal_alpha ${FOCAL_ALPHA} \
    --focal_gamma_pos ${FOCAL_GAMMA_POS} \
    --focal_gamma_neg ${FOCAL_GAMMA_NEG} \
    --augment \
    --augment_intensity ${AUGMENT_INTENSITY} \
    --num_augment ${NUM_AUGMENT} \
    --num_workers ${NUM_WORKERS} \
    --no_eval \
    --device ${DEVICE} \
    --use_dp \
    --gpu_ids 0 1
    "$@"

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: ${SAVE_DIR}"
echo "=========================================="
