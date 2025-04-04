#!/bin/bash

# Define variables
# PEFT_MODEL_PATH="/home/siyue/Projects/llm2vec_reason/output/supervised/Meta-Llama-3-8B-Instruct-mntp-supervised-new-v0.1/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-24_l-4096_bidirectional-True_e-1_s-42_w-300_lr-0.0001_lora_r-16/checkpoint-2498"
BATCH_SIZE=20

# torchrun --nproc_per_node=4 experiments/run_supervised.py train_configs/supervised/MetaLlama3.json

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "/home/siyue/Projects/llm2vec_reason/output/supervised/Meta-Llama-3-8B-Instruct-mntp-supervised-theoremqa-theorems-v1.0/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-10_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-400" \
#     --task_name "BrightTheoremqaTheorems" \
#     --output_dir "results/in_domain/supervised_theoremqa_theorems_v1.0_400" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "/home/siyue/Projects/llm2vec_reason/output/supervised/Meta-Llama-3-8B-Instruct-mntp-supervised-theoremqa-theorems-v1.0/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-10_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000" \
#     --task_name "BrightTheoremqaTheorems" \
#     --output_dir "results/in_domain/supervised_theoremqa_theorems_v1.0_1000" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "output/supervised/Meta-Llama-3-8B-Instruct-mntp-supervised-leetcode-v0.3/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-3_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-200" \
#     --task_name "BrightLeetcode" \
#     --output_dir "results/in_domain/supervised_leetcode_v1.0_200" \
#     --batch_size "$BATCH_SIZE"

python experiments/mteb_eval_custom.py \
    --peft_model_name_or_path "/home/siyue/Projects/llm2vec_reason/output/simcse/Meta-Llama-3-8B-Instruct-mntp-simcse-leetcode-v1.0-hard/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-10_s-42_w-50_lr-0.0001_lora_r-16/checkpoint-210" \
    --task_name "BrightLeetcode" \
    --output_dir "results/in_domain/simcse_leetcode_v1.0_hard_210" \
    --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "/home/siyue/Projects/llm2vec_reason/output/simcse/Meta-Llama-3-8B-Instruct-mntp-simcse-leetcode-v1.0/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-3_s-42_w-50_lr-0.0001_lora_r-16/checkpoint-741" \
#     --task_name "BrightLeetcode" \
#     --output_dir "results/in_domain/simcse_leetcode_v1.0_741" \
#     --batch_size "$BATCH_SIZE"

# # Run the command
# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightTheoremqaTheorems" \
#     --output_dir "results/all_domain/supervised_new_theoremqa_theorems_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightTheoremqaQuestions" \
#     --output_dir "results/all_domain/supervised_new_theoremqa_questions_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightAops" \
#     --output_dir "results/all_domain/supervised_new_aops_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightPony" \
#     --output_dir "results/all_domain/supervised_new_pony_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightLeetcode" \
#     --output_dir "results/all_domain/supervised_new_leetcode_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightStackOverflow" \
#     --output_dir "results/all_domain/supervised_new_stackoverflow_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightEconomics" \
#     --output_dir "results/all_domain/supervised_new_economics_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightBiology" \
#     --output_dir "results/all_domain/supervised_new_biology_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "/home/siyue/Projects/llm2vec_reason/output/simcse/Meta-Llama-3-8B-Instruct-mntp-simcse-new-v0.1/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-24_l-4096_bidirectional-True_e-1_s-42_w-300_lr-0.0001_lora_r-16/checkpoint-2498" \
#     --task_name "BrightBiology" \
#     --output_dir "results/all_domain/simcse_new_biology_2498" \
#     --batch_size "$BATCH_SIZE"

    