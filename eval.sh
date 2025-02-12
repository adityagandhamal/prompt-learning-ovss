#!/bin/sh

config=$1
gpus=$2
output=$3

export DETECTRON2_DATASETS="/scratch/jayant/aditya/CAT-Seg/datasets"

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

# MUSES ALL CONDITIONS 
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/scratch/jayant/aditya/CAT-Seg/datasets/muses.json" \
 DATASETS.TEST \(\"MUSES_SNOW\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS "ck_cocoop_case2_30KFT_50Kck/model_final.pth" \
 #MODEL.WEIGHTS "masked_frozen_decoder_ck/model_final.pth" \
 #MODEL.WEIGHTS "/scratch/jayant/Aniruddh/CAT-Seg/output_masking_checkpoints/30MFT_30K_50Kck/model_final.pth" \
 #MODEL.WEIGHTS "MASKING55_FT_30K/model_final.pth" \
 #MODEL.WEIGHTS "/scratch/jayant/Aniruddh/CAT-Seg/model_base.pth" \
 $opts

# #ADE20k-150
# python train_net.py --config $config \
#  --num-gpus $gpus \
#  --dist-url "auto" \
#  --eval-only \
#  OUTPUT_DIR $output/eval \
#  DATASETS.TEST \(\"ade20k_150_test_sem_seg\"\,\) \
#  TEST.SLIDING_WINDOW "True" \
#  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
#  MODEL.WEIGHTS "cocoop_case2/model_0034999.pth" \
#  #MODEL.WEIGHTS $output/model_final.pth \
#  $opts

# #ADE20k-847
# python train_net.py --config $config \
#  --num-gpus $gpus \
#  --dist-url "auto" \
#  --eval-only \
#  OUTPUT_DIR $output/eval \
#  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade847.json" \
#  DATASETS.TEST \(\"ade20k_full_sem_seg_freq_val_all\"\,\) \
#  TEST.SLIDING_WINDOW "True" \
#  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
#  MODEL.WEIGHTS $output/model_final.pth \
#  $opts

# #Pascal VOC
# python train_net.py --config $config \
#  --num-gpus $gpus \
#  --dist-url "auto" \
#  --eval-only \
#  OUTPUT_DIR $output/eval \
#  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/voc20.json" \
#  DATASETS.TEST \(\"voc_2012_test_sem_seg\"\,\) \
#  TEST.SLIDING_WINDOW "True" \
#  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
#  MODEL.WEIGHTS $output/model_final.pth \
#  $opts

# #Pascal VOC-b
# python train_net.py --config $config \
#  --num-gpus $gpus \
#  --dist-url "auto" \
#  --eval-only \
#  OUTPUT_DIR $output/eval \
#  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/voc20b.json" \
#  DATASETS.TEST \(\"voc_2012_test_background_sem_seg\"\,\) \
#  TEST.SLIDING_WINDOW "True" \
#  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
#  MODEL.WEIGHTS $output/model_final.pth \
#  $opts

# #Pascal Context 59
# python train_net.py --config $config \
#  --num-gpus $gpus \
#  --dist-url "auto" \
#  --eval-only \
#  OUTPUT_DIR $output/eval \
#  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON  "datasets/pc59.json" \
#  DATASETS.TEST \(\"context_59_test_sem_seg\"\,\) \
#  TEST.SLIDING_WINDOW "True" \
#  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
#  MODEL.WEIGHTS $output/model_final.pth \
#  $opts

# #Pascal Context 459
# python train_net.py --config $config \
#  --num-gpus $gpus \
#  --dist-url "auto" \
#  --eval-only \
#  OUTPUT_DIR $output/eval \
#  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/pc459.json" \
#  DATASETS.TEST \(\"context_459_test_sem_seg\"\,\) \
#  TEST.SLIDING_WINDOW "True" \
#  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
#  MODEL.WEIGHTS $output/model_final.pth \
#  $opts

# cat $output/eval/log.txt | grep copypaste