#!/usr/bin/env bash
DATASET = glint
LOG = "logs/${DATASET}_${NET}.`date +'%Y-%m-%d'`.txt"
export CUDA_VISIBLE_DEVICES='0,1,2,3'
python train_distillation.py --loss arcface --dataset glint  2>&1 | tee ${LOG}
