#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>train_selected_10k_resnet50_esm2_2560_addition_o128.log 2>&1

python train-triplet-addition.py \
	--training_data selected_10k \
	--model_name selected10k_addition_best \
	--out_dim 128 \
	--epoch 7000
