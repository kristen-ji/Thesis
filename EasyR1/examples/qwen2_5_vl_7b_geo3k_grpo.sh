#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# Load Hugging Face token for training
export HUGGINGFACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_TOKEN=$HUGGINGFACE_HUB_TOKEN
echo "✅ Training: HF Token loaded from cache"

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=4 \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=2 # 2 nodes, 4 gpus per node
