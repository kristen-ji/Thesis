#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --job-name=ThinkLite_MAB
#SBATCH --output=ThinkLite_MAB_%A_%a.log
#SBATCH --error=ThinkLite_MAB_%A_%a.error
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --array=0-15
#SBATCH --account=hk-project-pai00089

module load devel/cuda/12.9

if [ -f ~/.hf_token ]; then
    export HUGGINGFACE_HUB_TOKEN=$(cat ~/.hf_token)
    export HF_TOKEN=$HUGGINGFACE_HUB_TOKEN
fi

# Memory optimization for PyTorch - reduce fragmentation
# Based on https://docs.pytorch.org/docs/stable/notes/cuda.html#environment-variables
# expandable_segments: reduces fragmentation by allowing segments to grow
# max_split_size_mb: limits max allocation size to reduce fragmentation
# roundup_power2_divisions: rounds up allocations to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:2

export CUDA_VISIBLE_DEVICES=0

chunk_idx=$SLURM_ARRAY_TASK_ID
num_chunks=16

# LoRA reduces trainable parameters from ~7B to ~10-50M (100-700x reduction)
# This dramatically reduces optimizer state memory from ~28GB to ~40-200MB
python mab.py \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --gpu-id 0 \
  --num-chunks $num_chunks \
  --chunk-idx $chunk_idx \
  --batch-size 1 \
  --n-steps 1000 \
  --lr 1e-5 \
  --tau 0.1 \
  --tau-decay 0.999 \
  --predictor-path predictor.pt \
  --output-model-path qwen_mab_chunk_${chunk_idx}.pt \
  --use-lora \
  --use-8bit \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --max-length 256
