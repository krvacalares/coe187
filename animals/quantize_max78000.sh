#!/bin/sh
# Generated Log folder during the Training
LOG_DIRECTORY="../ai8x-training/logs/2025.11.06-190726"

# Define possible model files
QAT_MODEL="$LOG_DIRECTORY/qat_best.pth.tar"
BEST_MODEL="$LOG_DIRECTORY/best.pth.tar"

# Check which model file exists
if [ -f "$QAT_MODEL" ]; then
    echo "Quantizing QAT-trained model..."
    INPUT_MODEL="$QAT_MODEL"
    OUTPUT_MODEL="$LOG_DIRECTORY/qat_best-quantized.pth.tar"
elif [ -f "$BEST_MODEL" ]; then
    echo "QAT model not found — using best.pth.tar instead..."
    INPUT_MODEL="$BEST_MODEL"
    OUTPUT_MODEL="$LOG_DIRECTORY/best-quantized.pth.tar"
else
    echo "Error: No checkpoint file found in $LOG_DIRECTORY"
    exit 1
fi

# Quantization for Cats vs Dogs
python quantize.py "$INPUT_MODEL" "$OUTPUT_MODEL" --device MAX78000 -v "$@"

echo "Quantization completed!"
