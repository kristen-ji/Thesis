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

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin

# DO NOT import vllm here - it will run platform detection which fails
# Import vllm lazily in __init__ after CUDA is initialized

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float, limit_images: int = 0
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        # Apply image limiting only if limit_images > 0
        image_list = multi_modal_data["images"]
        if limit_images > 0 and len(image_list) > limit_images:
            image_list = image_list[:limit_images]  # Take only the first limit_images images
            
        for image in image_list:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        # Apply video limiting only if limit_images > 0
        video_list = multi_modal_data["videos"]
        if limit_images > 0 and len(video_list) > limit_images:
            video_list = video_list[:limit_images]  # Take only the first limit_images videos
            
        for video in video_list:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer  # Store tokenizer for later use
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        # WORKAROUND: Force TP=1 to avoid multi-node NCCL deadlocks with vLLM
        if config.tensor_parallel_size > 1:
            config.tensor_parallel_size = 1
            # With hybrid engine, vLLM and FSDP share GPU memory
            # 0.5 = 20GB for vLLM, 20GB for FSDP training - balanced allocation
            config.gpu_memory_utilization = 0.5
        
        # Reduce max_model_len to fit in available KV cache memory for hybrid engine
        # With 0.5 GPU util, KV cache is limited, so use 8192 max instead of model default (128000)
        if config.max_model_len is None or config.max_model_len > 8192:
            config.max_model_len = 8192

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        # Handle Qwen2.5-VL model configuration
        if "Qwen2.5-VL" in model_path:
            # Set environment variable to help with HF operations
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        
        # Prepare multimodal parameters for vLLM 0.9.2
        mm_kwargs = {}
        if processor is not None:  # only VLMs have processor
            # Apply limit_mm_per_prompt if limit_images > 0
            if config.limit_images > 0:
                mm_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}
            # Use mm_processor_kwargs for multimodal configuration
            mm_kwargs["mm_processor_kwargs"] = {}
            

        # Load HF token for vLLM and set environment variables
        try:
            with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                hf_token = f.read().strip()
                os.environ["HF_TOKEN"] = hf_token
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        except FileNotFoundError:
            pass
        
        # Set environment variables to help with Ray + vLLM compatibility
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # Force spawn method for Ray compatibility
        # NOTE: VLLM_USE_RAY_SPMD_WORKER is NOT needed since we use TP=1 (single GPU per instance)
        
        # CRITICAL: Initialize CUDA BEFORE importing vLLM to fix platform detection
        # vLLM's platform detection runs at import time and fails with NoneType error
        if torch.cuda.is_available():
            # Initialize CUDA context to ensure torch.version.cuda and related attrs are properly set
            _ = torch.cuda.current_device()
            _ = torch.cuda.device_count()
            # Create a test tensor to fully initialize CUDA
            _ = torch.zeros(1, device='cuda')
        else:
            raise RuntimeError("CUDA is not available. vLLM requires CUDA GPUs.")
        
        # Set critical environment variables that vLLM reads for device inference
        # These must be set BEFORE importing vLLM modules
        os.environ["VLLM_DEVICE"] = "cuda"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Ensure consistent device ordering
        
        # NOW import vLLM after CUDA is fully initialized
        import vllm
        from vllm import LLM, RequestOutput, SamplingParams
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform
        
        # Patch: Fix platform's is_async_output_supported to return False instead of raising
        # This ensures compatibility across different vLLM versions
        if not hasattr(current_platform, '_is_async_patched'):
            current_platform.__class__.is_async_output_supported = lambda self, enforce_eager: False
            current_platform._is_async_patched = True
        
        # Use Hugging Face model ID instead of local path to get correct architecture
        hf_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        # Synchronize all ranks before vLLM initialization
        if dist.is_initialized():
            dist.barrier()
        
        try:
            # vLLM 0.10.0+ has better stability and proper device detection
            # NOTE: Device parameter removed - relies on environment variables (VLLM_DEVICE, CUDA_VISIBLE_DEVICES)
            self.inference_engine = LLM(
                model=hf_model_id,  # Use HF model ID for correct architecture
                tokenizer=hf_model_id,  # Use HF tokenizer ID for correct tokenization
                skip_tokenizer_init=False,
                trust_remote_code=config.trust_remote_code,
                dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
                seed=config.seed,
                max_model_len=config.max_model_len,  # Explicitly set max_model_len for KV cache sizing
                max_seq_len_to_capture=config.max_model_len or config.prompt_length + config.response_length,
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                enforce_eager=config.enforce_eager,
                disable_custom_all_reduce=False,
            )
        except Exception:
            raise
        
        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": True,  # Enable detokenization for proper text output
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                            self.config.limit_images,  # Pass limit_images parameter
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            
            # Extract text outputs since detokenize=True
            response_texts = [output.text for completion in completions for output in completion.outputs]
            
            # Convert text back to token IDs for training using the tokenizer from the class
            response_ids = []
            for text in response_texts:
                # Tokenize the response text using the tokenizer passed to the class
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                response_ids.append(tokens)
            
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)