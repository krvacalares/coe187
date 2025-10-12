#!/bin/sh

# READ FIRST:
# This script file is design for workshop. This script file contains three (3) different traiing setup:
# 1. Cats vs Dogs
#     For this the epoch is set to 250 and have a learning rate of 0.001, the model is based on ai85cdnet and the device having set to defualt (MAX78000)
# 2. KWS
#     For this the epoch is set to 200 and have a learning rate of 0.001, the model is based on ai85kws20net and the device is set to default (MAX78000)
# 3. Face ID (Temporary)
#     For this the epoch is set to 100 and have a learning rate of 0.001, the model is based on ai85faceidnet and the device is set to default (MAX7800)
# 4. Emotion Recognition
#     For this the epoch is set to 300, batch size of 100, and have a learning rate of 0.001, the model is based on ai85cifaremotionrecog and the device is set to default (MAX7800)


# Training Script for Cats vs Dogs
python train.py --epochs 5 --optimizer Adam --lr 0.001 --wd 0 --deterministic \
--compress policies/schedule-catsdogs.yaml --model ai85cdnet --dataset cats_vs_dogs \
--confusion --param-hist --embedding --device MAX78000 --workers 0 --enable-tensorboard


# Training Scripts of Key Word Spotting (KWS)
# #python train.py --epochs 5 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule_kws20.yaml --model ai85kws20net --dataset KWS_20 --confusion --device MAX78000 "$@"

# Training Scripts for Face ID (temporary)
# #python train.py --epochs 5 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule-faceid.yaml --model ai85faceidnet --dataset FaceID --batch-size 100 --device MAX78000 --regression --print-freq 250 "$@"

# Training scripts for Cifar Emotion Recognition
#python train.py --deterministic --epochs 5 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar-nas.yaml --model ai85cifaremotionrecog --dataset emotion_recognition --device MAX78000 --batch-size 100 --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_late_cifar.yaml --confusion "$@"



# Common Template for train script:
# python train.py --epochs <insert no.> --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress <insert scheduler directory> --model <insert base model> --dataset <insert dataset> --batch-size <insert batch-size> --confusion --device MAX78000 "$@"

# For a much more detailed explanation for the command line argument, you can use --help or check the repository:
# https://github.com/MaximIntegratedAI/ai8x-synthesis

    