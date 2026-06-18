#!/bin/bash

# ==================== Paths ====================

pretrain_path=""
restart=False
checkpoint_root=""
restart_epoch=1
if_new_epoch=True

tr_data="path_to_frame_sampling.py"

save_dir=""
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
num_workers=12
gpu_num=8

use_amp=True
verbose=True
cls_loss="focal"

# ==================== Run ====================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_CACHE_DISABLE=1
export NCCL_TIMEOUT=3600
ulimit -n 65535

torchrun --nproc_per_node=${gpu_num} --master_port=29600  ../src/run_training.py \
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
    --cls_loss ${cls_loss} \
    --warmup_epochs ${warmup_epochs} \
    --scheduler_step_mode ${scheduler_step_mode} \
    $( [ "$use_amp" = "True" ] && echo "--use_amp" ) \
    $( [ "$verbose" = "True" ] && echo "--verbose" ) \
    $( [ "$restart" = "True" ] && echo "--restart" ) \
    $( [ "$save_model" = "True" ] && echo "--save_model" ) \
    $( [ "$if_new_epoch" = "True" ] && echo "--if_new_epoch" ) \
    --pretrain_path "${pretrain_path}" \
    --checkpoint_root ${checkpoint_root}
