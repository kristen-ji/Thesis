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
"""Utils for tokenization."""

from typing import Optional
import os

# Apply comprehensive patch BEFORE importing transformers
def patch_transformers_for_trust_remote_code():
    """Monkey patch transformers to ensure trust_remote_code=True is passed everywhere."""
    try:
        # Patch cached_files and cached_file to ensure trust_remote_code is passed
        try:
            from transformers.utils.hub import cached_files, cached_file
            original_cached_files = cached_files
            original_cached_file = cached_file
            
            def patched_cached_files(path_or_repo_id, filenames, **kwargs):
                if 'trust_remote_code' not in kwargs:
                    kwargs['trust_remote_code'] = True
                # Also ensure token is passed if not present
                if 'token' not in kwargs:
                    import os
                    token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
                    if token:
                        kwargs['token'] = token
                return original_cached_files(path_or_repo_id, filenames, **kwargs)
            
            def patched_cached_file(path_or_repo_id, filename, **kwargs):
                if 'trust_remote_code' not in kwargs:
                    kwargs['trust_remote_code'] = True
                # Also ensure token is passed if not present
                if 'token' not in kwargs:
                    import os
                    token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
                    if token:
                        kwargs['token'] = token
                return original_cached_file(path_or_repo_id, filename, **kwargs)
            
            # Replace the functions in transformers.utils.hub module
            import transformers.utils.hub as hub_module
            hub_module.cached_files = patched_cached_files
            hub_module.cached_file = patched_cached_file
            
        except Exception as e:
            print(f"⚠️ Could not patch cached_files/cached_file: {e}")
        
        # Also patch hf_hub_download directly as a fallback
        try:
            from huggingface_hub import hf_hub_download
            original_hf_hub_download = hf_hub_download
            
            def patched_hf_hub_download(repo_id, filename, **kwargs):
                # Don't pass trust_remote_code to hf_hub_download as it doesn't accept it
                # Just ensure token is passed if not present
                if 'token' not in kwargs:
                    import os
                    token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
                    if token:
                        kwargs['token'] = token
                return original_hf_hub_download(repo_id, filename, **kwargs)
            
            # Replace the function in huggingface_hub module
            import huggingface_hub
            huggingface_hub.hf_hub_download = patched_hf_hub_download
            
        except Exception as e:
            print(f"⚠️ Could not patch hf_hub_download: {e}")
        
        # Patch video processor loading as well
        try:
            from transformers.video_processing_utils import BaseVideoProcessor
            original_get_video_processor_dict = BaseVideoProcessor.get_video_processor_dict
            
            @classmethod
            def patched_get_video_processor_dict(cls, pretrained_model_name_or_path, **kwargs):
                if 'trust_remote_code' not in kwargs:
                    kwargs['trust_remote_code'] = True
                # Also ensure token is passed if not present
                if 'token' not in kwargs:
                    import os
                    token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
                    if token:
                        kwargs['token'] = token
                return original_get_video_processor_dict(pretrained_model_name_or_path, **kwargs)
            
            BaseVideoProcessor.get_video_processor_dict = patched_get_video_processor_dict
            
        except Exception as e:
            print(f"⚠️ Could not patch video processor: {e}")
        
        # Patch ImageProcessingMixin.get_image_processor_dict directly
        try:
            from transformers.image_processing_base import ImageProcessingMixin
            original_get_image_processor_dict = ImageProcessingMixin.get_image_processor_dict
            
            @classmethod
            def patched_get_image_processor_dict(cls, pretrained_model_name_or_path, **kwargs):
                if 'trust_remote_code' not in kwargs:
                    kwargs['trust_remote_code'] = True
                if 'token' not in kwargs:
                    import os
                    token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
                    if token:
                        kwargs['token'] = token
                return original_get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
            
            ImageProcessingMixin.get_image_processor_dict = patched_get_image_processor_dict
            
        except Exception as e:
            print(f"⚠️ Could not patch ImageProcessingMixin: {e}")
        
    except Exception as e:
        print(f"⚠️ Could not patch transformers in tokenizer.py: {e}")
        import traceback
        traceback.print_exc()

# Apply the patch immediately
print(f"🔧 Applying patches from process {os.getpid()}")
patch_transformers_for_trust_remote_code()

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin


def get_tokenizer(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if override_chat_template is not None:
        tokenizer.chat_template = override_chat_template

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor."""
    # Ensure trust_remote_code is set for vision models
    if 'trust_remote_code' not in kwargs:
        kwargs['trust_remote_code'] = True
    
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    if override_chat_template is not None:
        processor.chat_template = override_chat_template

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/auto/processing_auto.py#L386
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor
