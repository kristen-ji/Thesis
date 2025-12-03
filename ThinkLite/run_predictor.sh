#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --job-name=ThinkLite_PRED
#SBATCH --output=ThinkLite_PRED_%j.log
#SBATCH --error=ThinkLite_PRED_%j.error
#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --account=hk-project-pai00089
#SBATCH --priority=100

module load devel/cuda/12.9

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

if [ -f ~/.hf_token ]; then
    export HUGGINGFACE_HUB_TOKEN=$(cat ~/.hf_token)
    export HF_TOKEN=$HUGGINGFACE_HUB_TOKEN
fi

# Example: train predictor on one or more MCTS parquet/jsonl outputs
# Adjust paths according to what you actually saved
python predictor_train.py \
  --parquet-paths output_files/mcts_qwen_*.parquet \
  --min-iters-positive 5 \
  --batch-size 64 \
  --epochs 5 \
  --lr 1e-4 \
  --output-path predictor.pt