#!/bin/bash
time=`date +"%m-%d-%H-%M"`
if [ ! -d "checkpoint" ]; then
    mkdir checkpoint
fi
if [ ! -d "log" ]; then
    mkdir log
fi
LOG=log/test_pool_${time}.log

# dir path flags
python test_beam_search.py --workspace "/home/dagui/.keras/datasets/" \
    --feature_path "image_vgg19_fc2_feature.h5" --ifpool False \
	# --feature_path "image_vgg19_block5_pool_feature.h5" --ifpool True \
	--caption_file_path "{}.txt" \
    --model_weights "/home/dagui/Documents/class-slides/pattern recognization/course_project/checkpoint/weights.010-0.791.hdf5" \ 
	--save_path "." \
	--caption_len 30 \
	--embedding_size 256 \
    --image_embedding_size 128 \
	--RNN_out_units 512 \
	--batch_size 40 \  # no use in test
	--epochs 500 \   # no use in test
	--num_RNN_layers 3 \
	--RNN_category "LSTM" 2>&1|tee ${LOG}
#if finetune use this
#	--if_finetune=True \
#	--weight_dir="." \
