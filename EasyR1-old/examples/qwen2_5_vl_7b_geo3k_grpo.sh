#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# Essential memory optimizations only
export RAY_OBJECT_STORE_MEMORY=8000000000  # 8GB for multimodal processing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_MEMORY_FRACTION=0.8  # Limit GPU memory usage to 80%

# Reduce verbose logging
export VLLM_LOGGING_LEVEL=WARNING

# Fix Flash Attention compatibility issue
export FLASH_ATTENTION_FORCE_DISABLE=1


MODEL_PATH=/hkfs/work/workspace/scratch/st_st190232-myspace/models/Qwen2.5-VL-7B-Instruct  # local model path

python3.11 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.32 \
    worker.rollout.max_model_len=2048 \
    worker.rollout.max_num_batched_tokens=2048 \
    trainer.find_last_checkpoint=false \
    trainer.save_checkpoint_path=/hkfs/work/workspace/scratch/st_st190232-myspace/checkpoints