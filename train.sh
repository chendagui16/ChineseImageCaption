#!/bin/bash
time=`date +"%m-%d-%H-%M"`
if [ ! -d "checkpoint" ]; then
    mkdir checkpoint
fi
if [ ! -d "log" ]; then
    mkdir log
fi
LOG=log/train_pool_${time}.log

# dir path flags
python train.py --workspace "/home/dagui/.keras/datasets/" \
	--feature_path "image_vgg19_block5_pool_feature.h5" --ifpool True \
	--caption_file_path "{}.txt" \
	--save_path "." \
	--caption_len 30 \
	--embedding_size 256 \
	--RNN_out_units 512 \
	--batch_size 40 \
	--epochs 500 \
	--num_RNN_layers 3 \
	--RNN_category "LSTM" 2>&1|tee ${LOG}
#if finetune use this
#	--if_finetune=True \
#	--weight_dir="." \
