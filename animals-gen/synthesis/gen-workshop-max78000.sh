#!/bin/sh
DEVICE="MAX78000"
GEN_PATH="C:\MaximSDK\Examples\MAX78000\CNN\animals"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# Input files
QUANTIZED_MODEL="../ai8x-training/logs/2025.11.06-190726/qat_best-quantized.pth.tar"
YAML="networks/cats-dogs-hwc.yaml"
SAMPLE="C:\Users\Karl\Documents\Works\School\COE187.1\cats_dogs\ai8x-training\sample_animals.npy"

# Generate for Cats vs Dogs
python ai8xize.py --test-dir $GEN_PATH \
--prefix animals \
--overwrite \
--checkpoint-file $QUANTIZED_MODEL \
--config-file $YAML \
--sample-input $SAMPLE \
--softmax $COMMON_ARGS --fifo "$@"
