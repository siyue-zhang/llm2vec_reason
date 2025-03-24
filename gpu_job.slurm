#!/bin/bash
#SBATCH --job-name=llm2vec
#SBATCH --open-mode=append
#SBATCH --output=./monitor/leetcode.out
#SBATCH --error=./monitor/leetcode.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mem=200G
#SBATCH -c 4

singularity exec --nv --overlay /scratch/sz4651/Projects/UnifiedSKG/overlay-skg2.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c "
source ~/.bashrc \
&& conda activate llm2vec \
&& cd /scratch/sz4651/Projects/llm2vec_reason \
&& torchrun --nproc_per_node=4 experiments/run_supervised.py train_configs/supervised/MetaLlama3_hpc.json"