#!/bin/bash

# deepspeed --include="localhost:1,5,6" ../../llava/train/train.py \
#     --deepspeed  /research/video_metaphor/LLaVA/scripts/zero2.json\
#     --model_name_or_path lmsys/vicuna-13b-v1.5 \
#     --version plain \
#     --data_path /research/video_metaphor/LLaVA/playground/data/mscoco/pretraining_cocodata_true_caption.json \
#     --eval_data_path /research/video_metaphor/LLaVA/playground/data/mscoco/pretraining_cocodata_true_caption.json \
#     --image_folder ./playground/data/LLaVA-Pretrain/images \
#     --vision_tower microsoft/git-large-vatex \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-13b-self-pretrain-true-caption \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 5000 \
#     --save_total_limit 2 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb


deepspeed --include="localhost:6,7" ../../llava/train/train.py \
    --deepspeed  /research/video_metaphor/LLaVA/scripts/zero2.json\
    --model_name_or_path /research/video_metaphor/base_models/vicuna-13b-v1.5 \
    --version plain \
    --data_path /research/video_metaphor/LLaVA/playground/pretrain/ispy_metaphor_pretraining_data.json \
    --eval_data_path /research/video_metaphor/LLaVA/playground/pretrain/ispy_metaphor_pretraining_data.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower /research/video_metaphor/base_models/git-large-vatex \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --interval_type "image_combination"\
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /research/video_metaphor/LLaVA/checkpoints/llava-v1.5-13b-ispy-pretrain \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name 'ispy_pretraining'
