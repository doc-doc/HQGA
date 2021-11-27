#!/bin/bash
GPU=$1
MODE=$2
CUDA_VISIBLE_DEVICES=$GPU python main_qa.py \
	--mode $MODE
