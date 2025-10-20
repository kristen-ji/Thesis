# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os


import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # Load HF token in the Runner process
        try:
            with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                hf_token = f.read().strip()
                os.environ["HF_TOKEN"] = hf_token
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        except FileNotFoundError:
            print("Warning: HF token not found in Runner process")
        
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # Get HF token for direct passing to transformers
        hf_token_for_transformers = None
        try:
            with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                hf_token_for_transformers = f.read().strip()
        except FileNotFoundError:
            print("Warning: Could not load HF token for transformers")

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
            token=hf_token_for_transformers,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
            token=hf_token_for_transformers,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    # Load HF token for Ray workers
    hf_token = None
    try:
        with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
            hf_token = f.read().strip()
    except FileNotFoundError:
        print("Warning: HF token not found in ~/.cache/huggingface/token")
    
    # Set environment variables for current process
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",  # Reduce logging noise
                    "VLLM_LOGGING_LEVEL": "DEBUG",  # Enable verbose logging for device inference
                    "VLLM_DEVICE": "cuda",  # Force vLLM to use CUDA devices
                "PYTHONUNBUFFERED": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "8",  # Increased from 1 to prevent deadlocks
                # NCCL timeout and stability settings - CRITICAL for multi-node
                "NCCL_TIMEOUT": "1800",  # 30 minutes timeout for NCCL operations
                "NCCL_IB_TIMEOUT": "50",  # Increase InfiniBand timeout
                "TORCH_NCCL_BLOCKING_WAIT": "1",  # Use blocking wait for better error reporting
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",  # Enable async error handling
                # Memory optimization settings
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",  # Removed expandable_segments - incompatible with vLLM
                "CUDA_LAUNCH_BLOCKING": "0",
                "TORCH_CUDNN_V8_API_ENABLED": "1",
                # Hugging Face Hub settings for vision models
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TRANSFORMERS_OFFLINE": "0",
                "TRANSFORMERS_VERBOSITY": "error",
                # Force trust_remote_code for all transformers operations
                "TRANSFORMERS_SAFE_SERIALIZATION": "false",
                # Patch vLLM to use trust_remote_code
                "VLLM_USE_MODELSCOPE": "false",
                # Disable Ray log deduplication for better debugging
                "RAY_DEDUP_LOGS": "0",
                # Network stability improvements
                "RAY_DISABLE_IMPORT_WARNING": "1",
                "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": "1",
                "RAY_OBJECT_STORE_MEMORY": "8000000000",  # 8GB object store for multimodal processing
                "RAY_TASK_RETRY_DELAY": "5",  # Retry failed tasks after 5 seconds
                "RAY_TASK_MAX_RETRIES": "3",  # Retry up to 3 times
            }
        }
        
        # Propagate CUDA visibility from the launcher into Ray workers if present
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            runtime_env["env_vars"]["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        # Add HF token to runtime environment if available
        if hf_token:
            runtime_env["env_vars"]["HF_TOKEN"] = hf_token
            runtime_env["env_vars"]["HUGGINGFACE_HUB_TOKEN"] = hf_token
        else:
            # Set a dummy token to avoid authentication issues with public models
            runtime_env["env_vars"]["HUGGINGFACE_HUB_TOKEN"] = "hf_dummy_token_for_public_models"
            runtime_env["env_vars"]["HF_TOKEN"] = "hf_dummy_token_for_public_models"
        
        ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
