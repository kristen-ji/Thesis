#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# Memory optimization for 32B model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:16

# Ray memory optimization - increase object store for multimodal processing
export RAY_OBJECT_STORE_MEMORY=16000000000  # 16GB object store for 32B model
export RAY_DEDUP_LOGS=0
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1


# Additional memory optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_DISABLE_LOG_STATS=1
export VLLM_DISABLE_ERROR_DUMPS=1

# Suppress Triton compilation warnings
export TRITON_DISABLE_LINE_INFO=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export TRITON_COMPILE_CACHE_DIR=/tmp/triton_compile_cache

# Create Triton cache directories
mkdir -p /tmp/triton_cache
mkdir -p /tmp/triton_compile_cache

# Set temporary directories to remote workspace to reduce local storage pressure
export TMPDIR=/hkfs/work/workspace/scratch/st_st190232-myspace/temp_data
export TEMP=/hkfs/work/workspace/scratch/st_st190232-myspace/temp_data
export TMP=/hkfs/work/workspace/scratch/st_st190232-myspace/temp_data

# Create temporary directories if they don't exist
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/cache
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/checkpoints
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/temp_data

# Set Hugging Face cache to remote workspace
export HF_HOME=/hkfs/work/workspace/scratch/st_st190232-myspace/cache
export TRANSFORMERS_CACHE=/hkfs/work/workspace/scratch/st_st190232-myspace/cache
export HF_DATASETS_CACHE=/hkfs/work/workspace/scratch/st_st190232-myspace/cache

# Set Ray temporary directory to short path to avoid socket filename length limit
export RAY_TMPDIR=/tmp

# Connect to existing Ray cluster (set by 32B_GRPO.sh)
if [ -n "$ip_head" ] && [ -n "$redis_password" ]; then
    export RAY_ADDRESS=$ip_head
    export RAY_REDIS_PASSWORD=$redis_password
    echo "Connecting to existing Ray cluster at $ip_head"
else
    echo "Warning: Ray cluster variables not found. Make sure to run 32B_GRPO.sh first."
fi

# Load Hugging Face token for training
export HUGGINGFACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_TOKEN=$HUGGINGFACE_HUB_TOKEN

MODEL_PATH=Qwen/Qwen2.5-VL-32B-Instruct  # replace it with your local file path

python3.11 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=1 \
    worker.rollout.gpu_memory_utilization=0.3 \
    worker.actor.fsdp.enable_cpu_offload=true \
    trainer.experiment_name=qwen2_5_vl_32b_geo_grpo \
    trainer.save_checkpoint_path=/hkfs/work/workspace/scratch/st_st190232-myspace/checkpoints
