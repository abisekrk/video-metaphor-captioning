#!/bin/bash




deepspeed --include="localhost:0,1,2,3" ../../llava/train/train.py \
    --deepspeed  /research/video_metaphor/LLaVA/scripts/zero3.json\
    --model_name_or_path /research/video_metaphor/base_models/vicuna-13b-v1.5 \
    --version v1 \
    --data_path /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/train.json\
    --eval_data_path /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/val.json\
    --image_folder ../../playground/data/irfl \
    --vision_tower /research/video_metaphor/base_models/git-large-vatex \
    --pretrain_mm_mlp_adapter /research/video_metaphor/LLaVA/checkpoints/llava-v1.5-13b-ispy-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --interval_type "multi_interval"\
    --no_of_intervals 4\
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True\
    --output_dir ../../checkpoints/vmcd_ispy/llava1.5-13b-24f-4part\
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --load_best_model_at_end True \
    --report_to wandb \
    --run_name 'vmcd_llava_finetune_26f-4part_ispy'

# 8 Part folder contains checkpoint for 6 part training

# deepspeed --include="localhost:4,5,6,7" ../../llava/train/train.py \
#     --deepspeed  /research/video_metaphor/LLaVA/scripts/zero3.json\
#     --model_name_or_path lmsys/vicuna-13b-v1.5 \
#     --version v1 \
#     --data_path /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/train.json\
#     --eval_data_path /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/val.json\
#     --image_folder ../../playground/data/irfl \
#     --vision_tower microsoft/git-large-vatex \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -1 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length False \
#     --bf16 True\
#     --output_dir ../../checkpoints/vmcd_v2/llava1.5-13b-2parts\
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "epoch" \
#     --save_strategy "epoch" \
#     --save_steps 1000 \
#     --save_total_limit 2 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 1 \
#     --lazy_preprocess True \
#     --load_best_model_at_end True \
#     --report_to wandb \
#     --run_name 'vmcd_llava_finetune_2parts'



# --bf16 False \
# --load_best_model_at_end True \