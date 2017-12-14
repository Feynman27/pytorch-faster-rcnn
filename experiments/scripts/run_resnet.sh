#!/usr/bin/env bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python ./tools/trainval_net.py \
--weight data/pretrained_models/res101.pth \
--imdb voc_2007_trainval \
--imdbval voc_2007_test  \
--iters 80000 \
--cfg experiments/cfgs/res101.yml \
--net res101 \
--set ANCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1.0,2.0]
