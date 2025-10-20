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

import inspect
import re
from typing import Iterable, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel
from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from ...protocol import DataProto, all_gather_data_proto
from ...utils.fsdp_utils import load_fsdp_model, offload_fsdp_model
from ...utils.model_utils import print_gpu_memory_usage
from .base import BaseShardingManager


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        device_mesh: DeviceMesh,
        use_param_offload: bool,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.use_param_offload = use_param_offload
        self.loaded = False

        self.world_size = dist.get_world_size()
        # Handle TP=1 case where vLLM doesn't initialize TP groups
        try:
            self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
            self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
            self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group
        except (AssertionError, AttributeError):
            # TP=1 case: no tensor parallelism, treat as single GPU
            self.tp_size = 1
            self.tp_rank = 0
            self.tp_group = None

        # Record freed bytes to estimate memory usage correctly
        # https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
        self.freed_bytes = 0

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        gen_dp_rank = self.device_mesh["dp"].get_local_rank()
        torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)

    def _rename_weight_keys(self, actor_weights: dict[str, Union[torch.Tensor, DTensor]], model: PreTrainedModel):
        # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        original_weights = {}
        for key, value in actor_weights.items():
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, key)
                # Early exit of the loop
                if n_replace > 0:
                    break

            original_weights[key] = value

        return original_weights

    def _make_weight_iterator(
        self, actor_weights: dict[str, Union[torch.Tensor, DTensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in actor_weights.items():
            yield name, tensor.full_tensor() if self.world_size != 1 else tensor

    def _sync_weight_to_vllm(self):
        if self.use_param_offload:
            load_fsdp_model(self.module)

        actor_weights = get_model_state_dict(self.module)
        actor_weights = self._rename_weight_keys(actor_weights, self.module._fsdp_wrapped_module)
        print_gpu_memory_usage("After gather model weights in sharding manager")

        # With vLLM v0 engine, weight syncing during wake_up is fully supported
        # Sync weights from FSDP to vLLM to ensure consistent model state
        llm_engine = self.inference_engine.llm_engine
        
        # vLLM 0.8.3 uses v0 engine by default - perform proper weight syncing
        print("vLLM 0.8.3 with v0 engine - syncing weights from FSDP to vLLM")
        model = llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(self._make_weight_iterator(actor_weights))
        print("Weight sync completed successfully")

        del actor_weights
        if self.use_param_offload:
            offload_fsdp_model(self.module)

        torch.cuda.empty_cache()
        print_gpu_memory_usage("After sync model weights in sharding manager")

    def load_vllm_and_sync_weights(self):
        """Load vllm engine and sync model weights to vllm model."""
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()
        assert self.loaded is False, "vllm engine has already been loaded"
        self.loaded = True

        print_gpu_memory_usage("Before vllm wake up in sharding manager")
        
        # Add synchronization barrier to prevent multiple workers from initializing vLLM simultaneously
        if dist.is_initialized():
            dist.barrier()
        
        # Add error handling for vLLM wake_up with timeout
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("vLLM wake_up timed out after 300 seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 minute timeout
            
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()
            
            signal.alarm(0)  # Cancel the alarm
        except Exception as e:
            signal.alarm(0)  # Cancel the alarm in case of error
            raise

        self._sync_weight_to_vllm()

        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("vLLM wake_up with kv_cache timed out after 300 seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 minute timeout
            
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])
            else:
                self.inference_engine.wake_up()
            
            signal.alarm(0)  # Cancel the alarm
        except Exception as e:
            signal.alarm(0)  # Cancel the alarm in case of error
            raise

        print_gpu_memory_usage("After vllm wake up in sharding manager")
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def offload_vllm(self):
        """Offload vllm engine."""
        assert self.loaded is True, "vllm engine has not been loaded"
        self.loaded = False

        print_gpu_memory_usage("Before vllm offload in sharding manager")
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
        self.inference_engine.sleep(level=1)
        free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
        self.freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        print_gpu_memory_usage("After vllm offload in sharding manager")

        self.module.train()
        torch.cuda.empty_cache()  # add empty cache after each compute

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size > 1 and self.tp_group is not None:
            all_gather_data_proto(data, size=self.tp_size, group=self.tp_group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size > 1 and self.tp_group is not None:
            data = data.chunk(chunks=self.tp_size)[self.tp_rank]

        return data
