#!/bin/sh

config=$1
gpus=$2
output=$3

export DETECTRON2_DATASETS="/four_tb/ADITYA/CAT-Seg/datasets"
# gpu_ids="1" # coop
# gpu_ids="0,1" 
# export CUDA_VISIBLE_DEVICES=$gpu_ids

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --resume \
 OUTPUT_DIR $output \
 $opts

#sh eval.sh $config $gpus $output $opts