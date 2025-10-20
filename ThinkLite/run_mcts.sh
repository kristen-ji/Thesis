#!/bin/bash

#SBATCH --partition=accelerated
#SBATCH --job-name=ThinkLite_MCTS
#SBATCH --output=ThinkLite_MCTS_%A_%a.log
#SBATCH --time=48:00:00  # Extended time limit for 70k dataset

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16  # Increased CPU cores
#SBATCH --mem=100G  # Reduced memory allocation

#SBATCH --array=0-15  # Match the number of chunks (16 chunks)
#SBATCH --account=hk-project-pai00072
#SBATCH --priority=100

# Load modules
module load devel/cuda/12.9

# Set environment variables for optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

# Load Hugging Face token
if [ -f ~/.hf_token ]; then
    export HUGGINGFACE_HUB_TOKEN=$(cat ~/.hf_token)
    export HF_TOKEN=$HUGGINGFACE_HUB_TOKEN
fi

# Create output directory
mkdir -p output_files

# Get the array task ID (0-7)
chunk_idx=$SLURM_ARRAY_TASK_ID


# Check if CUDA is available and fix device mapping
if ! python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"; then
    echo "ERROR: CUDA not available in Python"
    exit 1
fi

# Fix CUDA device mapping - ensure we use the correct GPU
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU 0 for this job"

output_prefix="./output_files/mcts_qwen_"
num_chunks=16

# Run MCTS for this specific chunk with optimized parameters
python mcts.py \
    --output_file ${output_prefix}$((chunk_idx+1)).parquet \
    --num-chunks $num_chunks \
    --chunk-idx $chunk_idx \
    --gpu-id 0 \
    --max_num_iterations 8  # Optimized for speed vs quality balance