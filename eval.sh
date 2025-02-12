export DETECTRON2_DATASETS="/four_tb/ADITYA/CAT-Seg/datasets/"
gpu_ids="1"
export CUDA_VISIBLE_DEVICES=$gpu_ids

#ADE20k-150
python train_net.py --config configs/vitb_384.yaml \
 --num-gpus 1 \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR eval_checkpoints_og_cocoop80K \
 DATASETS.TEST \(\"coco_2017_test_stuff_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "False" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[2,2]" \
 MODEL.WEIGHTS output_catseg_cocoop80K/model_0069999.pth \
 $opts

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