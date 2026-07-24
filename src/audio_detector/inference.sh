#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# Configuration
PEAV_CHECKPOINT="./pe-av-base"
CHECKPOINT="./weights/audio_model.pth" 
INPUT_CSV="" # TODO
OUTPUT_CSV="" # TODO

DEVICE="cuda"
MODE="infer"

# Run inference
python inference.py \
    --peav-checkpoint ${PEAV_CHECKPOINT} \
    --checkpoint ${CHECKPOINT} \
    --input ${INPUT_CSV} \
    --output ${OUTPUT_CSV} \
    --device ${DEVICE} \
    --mode ${MODE} \
    --resume
