#!/bin/bash

#SBATCH --partition=accelerated
#SBATCH --job-name=ThinkLite_MCTS
#SBATCH --output=ThinkLite_MCTS_%A_%a.log
#SBATCH --time=08:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G

#SBATCH --array=0-7

# Load modules
module load devel/cuda/12.9

# Load Hugging Face token
if [ -f ~/.hf_token ]; then
    export HUGGINGFACE_HUB_TOKEN=$(cat ~/.hf_token)
    export HF_TOKEN=$HUGGINGFACE_HUB_TOKEN
fi


# Change to ThinkLite directory
#cd ThinkLite

# Create output directory
mkdir -p output_files

# Get the array task ID (0-7)
chunk_idx=$SLURM_ARRAY_TASK_ID

# Use GPU ID based on task ID
gpu_id=$((chunk_idx % 4))
#gpu_id=$chunk_idx

output_prefix="./output_files/mcts_qwen_"
num_chunks=8

# Run MCTS for this specific chunk
python mcts.py \
    --output_file ${output_prefix}$((chunk_idx+1)).parquet \
    --num-chunks $num_chunks \
    --chunk-idx $chunk_idx \
    --gpu-id $gpu_id