#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_video \
    --model-path /research/video_metaphor/LLaVA/checkpoints/classification/llava-v1.5-13b_pt_finetune_18f_interval/checkpoint-159\
    --question-file /research/video_metaphor/LLaVA/playground/data/video_data/classification_data/test_questions.jsonl \
    --answers-file /research/video_metaphor/LLaVA/results/llava13-b/classification/18frames/test_interval_3ep_t_0_5.jsonl \
    --vision-tower microsoft/git-large-vatex \
    --temperature 0.5 \
    --conv-mode vicuna_v1

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_video \
    --model-path /research/video_metaphor/LLaVA/checkpoints/classification/llava-v1.5-13b_pt_finetune_18f_interval/checkpoint-159\
    --question-file /research/video_metaphor/LLaVA/playground/data/video_data/classification_data/test_questions.jsonl \
    --answers-file /research/video_metaphor/LLaVA/results/llava13-b/classification/18frames/test_interval_3ep_t_1.jsonl \
    --vision-tower microsoft/git-large-vatex \
    --temperature 1 \
    --conv-mode vicuna_v1

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_video \
    --model-path /research/video_metaphor/LLaVA/checkpoints/classification/llava-v1.5-13b_pt_finetune_18f_interval/checkpoint-265\
    --question-file /research/video_metaphor/LLaVA/playground/data/video_data/classification_data/test_questions.jsonl \
    --answers-file /research/video_metaphor/LLaVA/results/llava13-b/classification/18frames/test_interval_5ep_t_0.jsonl \
    --vision-tower microsoft/git-large-vatex \
    --temperature 0 \
    --conv-mode vicuna_v1

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_video \
    --model-path /research/video_metaphor/LLaVA/checkpoints/classification/llava-v1.5-13b_pt_finetune_18f_interval/checkpoint-265\
    --question-file /research/video_metaphor/LLaVA/playground/data/video_data/classification_data/test_questions.jsonl \
    --answers-file /research/video_metaphor/LLaVA/results/llava13-b/classification/18frames/test_interval_5ep_t_0_5.jsonl \
    --vision-tower microsoft/git-large-vatex \
    --temperature 0.5 \
    --conv-mode vicuna_v1

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_video \
    --model-path /research/video_metaphor/LLaVA/checkpoints/classification/llava-v1.5-13b_pt_finetune_18f_interval/checkpoint-265\
    --question-file /research/video_metaphor/LLaVA/playground/data/video_data/classification_data/test_questions.jsonl \
    --answers-file /research/video_metaphor/LLaVA/results/llava13-b/classification/18frames/test_interval_5ep_t_1.jsonl \
    --vision-tower microsoft/git-large-vatex \
    --temperature 1 \
    --conv-mode vicuna_v1


# deepspeed --include="localhost:6,7" ../../../llava/eval/model_vqa_video.py \
#     --deepspeed  /research/video_metaphor/LLaVA/scripts/zero2.json\
#     --model-path /research/video_metaphor/LLaVA/checkpoints/llava-v1.5-13b_video_wo_cHeader \
#     --question-file /research/video_metaphor/LLaVA/playground/data/video_data/video300_test_questions.jsonl \
#     --answers-file /research/video_metaphor/LLaVA/results/llava13-b/video_answers_test2.jsonl \
#     --vision-tower microsoft/git-large-vatex \
#     --temperature 0 \
#     --conv-mode vicuna_v1
