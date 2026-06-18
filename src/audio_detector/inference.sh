#!/bin/bash
# Inference script for CSV input
export CUDA_VISIBLE_DEVICES=7
# Configuration
CHECKPOINT="./weights/audio_model.20.pt"
INPUT_CSV=""
OUTPUT_CSV="audio_prob.csv"

DEVICE="cuda"
MODE="infer"

# Run inference
python inference.py \
    --checkpoint ${CHECKPOINT} \
    --input ${INPUT_CSV} \
    --output ${OUTPUT_CSV} \
    --save_features \
    --device ${DEVICE} \
    --mode ${MODE} \
    --resume
