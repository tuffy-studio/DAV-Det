#!/bin/bash

# ==================== Paths ====================

backbone_configure="./dinov3-vitl16-pretrain-lvd1689m"
tr_data= # TODO: set the path to your training data
use_lora=True
lora_r=32
lora_alpha=16
lora_dropout=0.1
use_deep_supervision=False
layer_indices="24"
unfreeze_norm=True
pretrain_path=""
img_size=512 # 512 for MVAD, 224 for Fakeavceleb

restart=False
checkpoint_root=""
restart_epoch=1
if_new_epoch=True


save_dir="./fakeavceleb_full"
save_model=True

mkdir -p $save_dir
mkdir -p ${save_dir}/models

# ==================== Training Params ====================

lr=1e-4
head_lr_ratio=1
token_head_lr_ratio=1
n_epochs=20
warmup_epochs=2
scheduler_step_mode="batch"
batch_size=112
accumulation_steps=16
num_workers=8
gpu_num=8

use_amp=True
verbose=True
cls_loss="focal"
focal_alpha=0.6 # 0.6 for MVAD, 0.4 for Fakeavceleb
focal_gamma_pos=2.0
focal_gamma_neg=2.0

# ==================== Run ====================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_CACHE_DISABLE=1
export NCCL_TIMEOUT=3600
ulimit -n 65535

torchrun --nproc_per_node=${gpu_num} --master_port=29600  ./run_training.py \
    --data_train ${tr_data} \
    --save_dir ${save_dir} \
    --lr ${lr} \
    --head_lr_ratio ${head_lr_ratio} \
    --token_head_lr_ratio ${token_head_lr_ratio} \
    --n_epochs ${n_epochs} \
    --restart_epoch ${restart_epoch} \
    --batch_size ${batch_size} \
    --accumulation_steps ${accumulation_steps} \
    --num_workers ${num_workers} \
    --gpu_num ${gpu_num} \
    --img_size ${img_size} \
    --cls_loss ${cls_loss} \
    --focal_alpha ${focal_alpha} \
    --focal_gamma_pos ${focal_gamma_pos} \
    --focal_gamma_neg ${focal_gamma_neg} \
    --warmup_epochs ${warmup_epochs} \
    --scheduler_step_mode ${scheduler_step_mode} \
    $( [ "$use_amp" = "True" ] && echo "--use_amp" ) \
    $( [ "$verbose" = "True" ] && echo "--verbose" ) \
    $( [ "$restart" = "True" ] && echo "--restart" ) \
    $( [ "$save_model" = "True" ] && echo "--save_model" ) \
    $( [ "$if_new_epoch" = "True" ] && echo "--if_new_epoch" ) \
    $( [ -n "${pretrain_path}" ] && echo "--pretrain_path ${pretrain_path}" ) \
    $( [ -n "${checkpoint_root}" ] && echo "--checkpoint_root ${checkpoint_root}" ) \
    --backbone_configure ${backbone_configure} \
    --layer_indices ${layer_indices} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    $( [ "$use_lora" = "True" ] && echo "--use_lora" || echo "--no_lora" ) \
    $( [ "$use_deep_supervision" = "True" ] && echo "--use_deep_supervision" || echo "--no_deep_supervision" ) \
    $( [ "$unfreeze_norm" = "True" ] && echo "--unfreeze_norm" || echo "--freeze_norm" )
