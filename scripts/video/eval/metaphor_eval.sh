#!/bin/bash





# CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_video \
#     --model-path /research/video_metaphor/LLaVA/checkpoints/vmcd_v2/llava1.5-13b-2parts/checkpoint-75\
#     --question-file /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/test_questions.jsonl \
#     --answers-file /research/video_metaphor/LLaVA/results/llava13-b/vmcd/2_parts/1_ep/answers_t_0_test.jsonl \
#     --vision-tower microsoft/git-large-vatex \
#     --temperature 0 \
#     --conv-mode vicuna_v1


# CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_video \
#     --model-path /research/video_metaphor/LLaVA/checkpoints/vmcd_ispy/llava1.5-13b-24f-4part/checkpoint-75\
#     --question-file /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/test_questions.jsonl \
#     --answers-file /research/video_metaphor/LLaVA/results/llava13-b/vmcd_ispy/24f_4parts/1_ep/answers_t_0_2_test.jsonl \
#     --vision-tower /research/video_metaphor/base_models/git-large-vatex \
#     --temperature 0.2 \
#     --interval-type "multi_interval"\
#     --no_of_intervals 4\
#     --conv-mode vicuna_v1

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_video \
    --model-path /research/video_metaphor/LLaVA/checkpoints/vmcd_ispy/llava1.5-13b-6f-1part/git-llava\
    --question-file /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/test_questions.jsonl \
    --answers-file /research/video_metaphor/LLaVA/results/llava13-b/vmcd_ispy/test_inference.jsonl \
    --vision-tower /research/video_metaphor/base_models/git-large-vatex\
    --temperature 0.2 \
    --interval-type "multi_interval"\
    --no_of_intervals 4\
    --conv-mode vicuna_v1





# CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_video \
#     --model-path /research/video_metaphor/LLaVA/checkpoints/vmcd_v2/llava1.5-13b-24f/checkpoint-375\
#     --question-file /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/test_questions.jsonl \
#     --answers-file /research/video_metaphor/LLaVA/results/llava13-b/vmcd/24_frames/5_ep/answers_t_0_test.jsonl \
#     --vision-tower microsoft/git-large-vatex \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_video \
#     --model-path /research/video_metaphor/LLaVA/checkpoints/vmcd_v2/llava1.5-13b-24f/checkpoint-375\
#     --question-file /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/test_questions.jsonl \
#     --answers-file /research/video_metaphor/LLaVA/results/llava13-b/vmcd/24_frames/5_ep/answers_t_0_2_test.jsonl \
#     --vision-tower microsoft/git-large-vatex \
#     --temperature 0.2 \
#     --conv-mode vicuna_v1


# ======================================================
# No pretrained model
# ======================================================


# CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_video \
#     --model-path /research/video_metaphor/LLaVA/checkpoints/vmcd_v2/llava1.5-13b-6f-no-pretrain/checkpoint-75\
#     --question-file /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/test_questions.jsonl \
#     --answers-file /research/video_metaphor/LLaVA/results/llava13-b/vmcd/no_pretrain/1_ep/answers_t_0_test.jsonl \
#     --vision-tower microsoft/git-large-vatex \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_video \
#     --model-path /research/video_metaphor/LLaVA/checkpoints/vmcd_v2/llava1.5-13b-6f-no-pretrain/checkpoint-75\
#     --question-file /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/test_questions.jsonl \
#     --answers-file /research/video_metaphor/LLaVA/results/llava13-b/vmcd/no_pretrain/1_ep/answers_t_0_2_test.jsonl \
#     --vision-tower microsoft/git-large-vatex \
#     --temperature 0.2 \
#     --conv-mode vicuna_v1


# CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_video \
#     --model-path /research/video_metaphor/LLaVA/checkpoints/vmcd_v2/llava1.5-13b-6f-no-pretrain/checkpoint-375\
#     --question-file /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/test_questions.jsonl \
#     --answers-file /research/video_metaphor/LLaVA/results/llava13-b/vmcd/no_pretrain/5_ep/answers_t_0_test.jsonl \
#     --vision-tower microsoft/git-large-vatex \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_video \
#     --model-path /research/video_metaphor/LLaVA/checkpoints/vmcd_v2/llava1.5-13b-6f-no-pretrain/checkpoint-375\
#     --question-file /research/video_metaphor/LLaVA/playground/data/video_data/vmcd_v2/test_questions.jsonl \
#     --answers-file /research/video_metaphor/LLaVA/results/llava13-b/vmcd/no_pretrain/5_ep/answers_t_0_2_test.jsonl \
#     --vision-tower microsoft/git-large-vatex \
#     --temperature 0.2 \
#     --conv-mode vicuna_v1
