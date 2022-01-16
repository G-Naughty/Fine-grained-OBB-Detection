#!/usr/bin/env bash

MODEL_NAME=$1
GPU_ID=$2
CLASSES=$3

python tools/train.py --model_name ${MODEL_NAME} --gpu-ids ${GPU_ID}

CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/test.py --model_name ${MODEL_NAME}

python result2gaofentype/pkl2txt.py --model_name ${MODEL_NAME} --classes ${CLASSES}

python result2gaofentype/txt2xml.py --model_name ${MODEL_NAME}

