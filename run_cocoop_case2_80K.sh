# 50K->(5K mask+cocoop)->25K cocoop case2 
export DETECTRON2_DATASETS="/four_tb/ADITYA/CAT-Seg/datasets/"
gpu_ids="1"
export CUDA_VISIBLE_DEVICES=$gpu_ids

#stage1 (5K cocoop) toggle
python train_net.py --config configs/vitb_384.yaml \
 --num-gpus 1 \
 --dist-url "auto" \
 --resume \
 OUTPUT_DIR output_cocoop_case2_80K/ \
 MODEL.SEM_SEG_HEAD.ENABLE_COCOOP "True" \
 MODEL.SEM_SEG_HEAD.CLIP_FINETUNE "coop" \
 SOLVER.MAX_ITER "80000" \
 TEST.EVAL_PERIOD "5000" \
 $opts
