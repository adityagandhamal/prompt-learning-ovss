# 50K->(5K mask+cocoop)->25K ogcase3 
export DETECTRON2_DATASETS="/four_tb/ADITYA/CAT-Seg/datasets/"
gpu_ids="0,1"
export CUDA_VISIBLE_DEVICES=$gpu_ids

#stage1 (5K cocoop) toggle
python train_net.py --config configs/vitb_384.yaml \
 --num-gpus 2 \
 --dist-url "auto" \
 OUTPUT_DIR output_cocoop_toggle5K/stage1_cocoop5K \
 MODEL.WEIGHTS /four_tb/ADITYA/CAT-Seg/out_iter50K/model_final.pth \
 MODEL.SEM_SEG_HEAD.ENABLE_COCOOP "True" \
 MODEL.SEM_SEG_HEAD.CLIP_FINETUNE "coop" \
 SOLVER.MAX_ITER "5000" \
 TEST.EVAL_PERIOD "5500" \
 $opts

# #stage2 -> 25K
# python train_net.py --config configs/vitb_384.yaml \
#  --num-gpus 2 \
#  --dist-url "auto" \
#  OUTPUT_DIR output_cocoop_toggle5K/stage2_25K \
#  MODEL.WEIGHTS output_cocoop_toggle5K/stage1_cocoop5K/model_final.pth \
#  MODEL.SEM_SEG_HEAD.ENABLE_COCOOP "False" \
#  MODEL.SEM_SEG_HEAD.CLIP_FINETUNE "attention" \
#  SOLVER.MAX_ITER "25000" \
#  TEST.EVAL_PERIOD "5000" \
#  $opts
