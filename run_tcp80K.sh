export DETECTRON2_DATASETS="/four_tb/ADITYA/CAT-Seg/datasets"

gpu_ids="1"
export CUDA_VISIBLE_DEVICES=$gpu_ids

python train_net.py --config configs/vitb_384.yaml \
 --num-gpus 1 \
 --dist-url "auto" \
 OUTPUT_DIR tcp80K_tcp_ft_clip_ft/ \
 MODEL.MASKING_RATIO "None" \
 MODEL.SEM_SEG_HEAD.ENABLE_TCP "True" \
 MODEL.SEM_SEG_HEAD.TCP_FINETUNE "True" \
 MODEL.SEM_SEG_HEAD.CLIP_FINETUNE "attention" \
 SOLVER.MAX_ITER "80000" \
 TEST.EVAL_PERIOD "5000" \
 $opts

