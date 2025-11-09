#!/bin/sh
# Evaluation script for Animals

MODEL="ai85cdnet"
DATASET="animals"
QUANTIZED_MODEL="../ai8x-training/logs/2025.11.06-190726/qat_best-quantized.pth.tar"

# Run evaluation
python train.py \
  --arch "$MODEL" \
  --dataset "$DATASET" \
  --confusion \
  --evaluate \
  --exp-load-weights-from "$QUANTIZED_MODEL" \
  --8-bit-mode \
  --save-sample 1 \
  --device MAX78000 "$@"
