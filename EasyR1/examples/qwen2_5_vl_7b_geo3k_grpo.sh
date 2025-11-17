#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/hkfs/home/project/hk-project-pai00012/st_st190232/use-ray-with-slurm/EasyR1/output.parquet \
    data.val_files=/hkfs/work/workspace/scratch/st_st190232-myspace/thinklite_hard_val.parquet \
    data.image_dir=/hkfs/work/workspace/scratch/st_st190232-myspace/thinklite_images \
    data.image_key=images \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.rollout_batch_size=256 \
    data.mini_rollout_batch_size=128 \
    worker.actor.global_batch_size=64 \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=true \
    worker.rollout.gpu_memory_utilization=0.38 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=mcts_qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=10 \
    trainer.find_last_checkpoint=false
