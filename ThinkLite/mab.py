import os
import math
import json
import io
import random
import argparse
import gc

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset

import open_clip

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except (ImportError, AttributeError):
    BITSANDBYTES_AVAILABLE = False
except Exception:
    BITSANDBYTES_AVAILABLE = False

# Check if FlashAttention2 is available
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
except Exception:
    FLASH_ATTN_AVAILABLE = False

import pdb

# ------------------------------------------------
# Helpers for chunking / IO
# ------------------------------------------------

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    print(f"DEBUG: Total data length: {len(lst)}")
    print(f"DEBUG: Number of chunks: {n}")
    print(f"DEBUG: Requested chunk index: {k}")
    print(f"DEBUG: Available chunk indices: 0-{len(chunks)-1}")
    print(f"DEBUG: Chunk sizes: {[len(chunk) for chunk in chunks]}")

    if k >= len(chunks):
        raise IndexError(f"Chunk index {k} out of range. Available chunks: 0-{len(chunks)-1}")
    return chunks[k]


def dump_to_jsonl(obj, path: str):
    with open(path, "w") as f:
        for x in obj:
            f.write(json.dumps(x) + "\n")


# ------------------------------------------------
# Dataset: one arm per sample
# ------------------------------------------------

class VQADatasetBytes(Dataset):
    """
    Wraps ThinkLite-VL examples; each item is a per-sample arm.
    Expects each record to have:
      - 'image'
      - 'problem' (contains <image>)
      - 'answer'
    """

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        d = self.records[idx]
        image_raw = d["image"]
        problem = d["problem"]
        answer = d["answer"]

        question = problem.split("<image>")[1]

        return {
            "idx": idx,
            "image": image_raw,
            "question": question,
            "answer": answer,
        }


def collate_fn(examples):
    return {
        "idxs": torch.tensor([e["idx"] for e in examples], dtype=torch.long),
        "images": [e["image"] for e in examples],
        "questions": [e["question"] for e in examples],
        "answers": [e["answer"] for e in examples],
    }


# ------------------------------------------------
# CLIP embedder
# ------------------------------------------------

class CLIPEmbedder:
    """CLIP-based embedding for multimodal input using OpenCLIP"""

    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        print(f"Initializing CLIP embedder: {model_name} with {pretrained}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"CLIP embedder initialized on {self.device}")

    def embed(self, image_raw, text: str):
        """
        image_raw: bytes OR PIL.Image.Image
        text: question string
        Returns: combined normalized [1, D] tensor on self.device
        """
        if isinstance(image_raw, Image.Image):  
            image = image_raw
        else:
            image = Image.open(io.BytesIO(image_raw)).convert("RGB")

        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            image_features = F.normalize(image_features, p=2, dim=-1)

            text_input = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(text_input)
            text_features = F.normalize(text_features, p=2, dim=-1)

            combined = torch.cat([image_features, text_features], dim=-1)
            combined = F.normalize(combined, p=2, dim=-1)

        return combined  # (1, D)


# ------------------------------------------------
# Predictor on CLIP embeddings (unchanged)
# ------------------------------------------------

class SimpleMLPPredictor(nn.Module):
    def __init__(self, emb_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # usability in [0,1]
        )

    def forward(self, x):
        return self.net(x)


def compute_clip_embeddings_and_predictor_scores(
    dataset,
    clip_embedder,
    predictor,
    batch_size=32,
    device="cuda",
):
    predictor.to(device)
    predictor.eval()

    all_embs = []
    all_scores = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for batch in tqdm(loader, desc="CLIP+predictor scoring"):
        batch_embs = []
        for img_raw, q in zip(batch["images"], batch["questions"]):
            emb = clip_embedder.embed(img_raw, q)  # (1, D)
            batch_embs.append(emb)

        batch_embs = torch.cat(batch_embs, dim=0).to(device)  # (B, D)

        with torch.no_grad():
            scores = predictor(batch_embs).squeeze(-1)  # (B,)

        all_embs.append(batch_embs.detach().cpu())
        all_scores.append(scores.detach().cpu())

    clip_embs = torch.cat(all_embs, dim=0).numpy()
    predictor_scores = torch.cat(all_scores, dim=0).numpy()
    return clip_embs, predictor_scores


# ------------------------------------------------
# Softmax (Boltzmann) bandit
# ------------------------------------------------

class SoftmaxBandit:
    def __init__(self, n_arms, prior_mean=None, tau=0.1, tau_decay=None, min_prob=1e-8):
        """
        n_arms: number of samples
        prior_mean: np.array[n_arms], e.g. predictor-based usability in [0,1]
        tau: initial temperature (higher = more exploration)
        tau_decay: multiplicative decay per step (e.g. 0.999). If None, keep tau constant.
        min_prob: floor probability to avoid numerical issues
        """
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float32)  # Q_i
        self.step = 0
        self.tau = tau
        self.tau_decay = tau_decay
        self.min_prob = min_prob

        if prior_mean is not None:
            p = prior_mean.astype(np.float32)
            p = (p - p.min()) / (p.max() - p.min() + 1e-8)  # normalize
            self.values[:] = p  # initialize Q_i from predictor

    def _current_tau(self):
        if self.tau_decay is None:
            return self.tau
        return max(self.tau * (self.tau_decay ** self.step), 1e-4)

    def _softmax_probs(self, available_mask=None):
        tau = self._current_tau()
        q = self.values.astype(np.float64).copy()

        if available_mask is not None:
            q = np.where(available_mask, q, -np.inf)

        max_q = np.nanmax(q)
        logits = (q - max_q) / max(tau, 1e-6)

        if np.all(~np.isfinite(logits)):
            probs = np.ones(self.n_arms, dtype=np.float64) / self.n_arms
        else:
            exp_logits = np.exp(logits)
            probs = exp_logits / (exp_logits.sum() + 1e-12)

        if available_mask is not None:
            probs = np.where(available_mask, probs, 0.0)

        probs = probs + self.min_prob
        probs = probs / probs.sum()
        return probs

    def select_batch(self, k, available_mask=None):
        self.step += 1
        probs = self._softmax_probs(available_mask=available_mask)
        k = min(k, self.n_arms)
        chosen = np.random.choice(
            np.arange(self.n_arms),
            size=k,
            replace=False,
            p=probs,
        )
        return chosen

    def update(self, chosen_indices, reward):
        if np.isscalar(reward):
            rewards = np.full(len(chosen_indices), reward, dtype=np.float32)
        else:
            rewards = np.asarray(reward, dtype=np.float32)

        for idx, r in zip(chosen_indices, rewards):
            self.counts[idx] += 1
            n = self.counts[idx]
            q = self.values[idx]
            self.values[idx] = q + (r - q) / n  # incremental mean


# ------------------------------------------------
# Qwen inputs & per-sample loss
# ------------------------------------------------

def build_qwen_inputs(processor, batch, device, max_length=1024):
    images = []
    for img_raw in batch["images"]:
        if isinstance(img_raw, Image.Image):
            img = img_raw
        else:
            img = Image.open(io.BytesIO(img_raw)).convert("RGB")
        images.append(img)

    questions = batch["questions"]
    answers = batch["answers"]

    # Use chat template format with image placeholders (required for Qwen2.5-VL)
    # This ensures image tokens are properly inserted in the text
    prompts = []
    prompt_lengths = []
    
    for q, a in zip(questions, answers):
        prompt_text = f"Question: {q}\nAnswer:"
        
        # Create conversation format with image placeholder
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }]
        
        # Apply chat template to get text with image tokens inserted
        text_with_image_tokens = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        
        # Append the answer to the prompt
        full_text = text_with_image_tokens + " " + a
        prompts.append(full_text)
        
        # Store prompt length (with image tokens) for label masking
        prompt_tokens = processor.tokenizer(text_with_image_tokens, return_tensors="pt", add_special_tokens=False)
        prompt_lengths.append(prompt_tokens.input_ids.size(1))

    # Process with images - processor will match image tokens with image features
    # Truncate to max_length to reduce memory usage during backward pass
    # Activation memory scales quadratically with sequence length
    proc_inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = proc_inputs["input_ids"]
    labels = input_ids.clone()

    # Create labels: mask the prompt part (including image tokens), keep only the answer
    for i, prompt_len in enumerate(prompt_lengths):
        if prompt_len < labels.size(1):
            labels[i, :prompt_len] = -100  # ignore prompt in loss
        # Answer part (after prompt_len) is kept for loss calculation

    proc_inputs["labels"] = labels
    return proc_inputs


def compute_per_sample_loss(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    vocab_size = logits.size(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    )
    loss_flat = loss_flat.view(shift_labels.size(0), -1)

    mask = (shift_labels != -100).float()
    loss_flat = loss_flat * mask

    losses_per_sample = loss_flat.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return losses_per_sample, outputs


# ------------------------------------------------
# MAB-guided training loop (softmax + local loss reduction)
# ------------------------------------------------

def train_with_mab_clip(
    model,
    processor,
    train_dataset,
    clip_embedder,
    predictor,
    batch_size=8,
    n_steps=1000,
    lr=1e-5,
    tau=0.1,
    tau_decay=0.999,
    device="cuda",
    max_length=1024,
):
    model.to(device)
    model.train()
    
    # Verify that model has trainable parameters (critical for 8-bit + LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found! Check that LoRA is properly applied or model is not frozen.")
    print(f"Found {len(trainable_params)} trainable parameter tensors")
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total_trainable:,}")
    
    # Check if model is quantized (8-bit quantization may not support gradient checkpointing)
    is_quantized = hasattr(model, 'hf_quantizer') or any(
        hasattr(module, 'weight') and hasattr(module.weight, 'SCB') 
        for module in model.modules()
    )
    
    # Enable gradient checkpointing to save memory (trades compute for memory)
    # Note: Gradient checkpointing may not work with 8-bit quantized models
    if hasattr(model, 'gradient_checkpointing_enable') and not is_quantized:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled to save memory")
    elif is_quantized:
        print("⚠️  Gradient checkpointing disabled for quantized model (not compatible with 8-bit quantization)")
    else:
        print("⚠️  Gradient checkpointing not available for this model")

    # 1) CLIP embeddings + predictor scores
    clip_embs, predictor_scores = compute_clip_embeddings_and_predictor_scores(
        dataset=train_dataset,
        clip_embedder=clip_embedder,
        predictor=predictor,
        batch_size=batch_size,
        device=device,
    )

    n_samples = len(train_dataset)
    bandit = SoftmaxBandit(
        n_arms=n_samples,
        prior_mean=predictor_scores,
        tau=tau,
        tau_decay=tau_decay,
    )

    # Aggressively free memory before creating optimizer (optimizer states require ~2x model params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Force garbage collection to free Python objects
        gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU memory before optimizer creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

    # Get only trainable parameters for optimizer (critical for 8-bit quantized models)
    # With 8-bit quantization, only LoRA adapter parameters should be trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found for optimizer! This usually means LoRA was not properly applied.")
    
    # Try to use fused AdamW if available (more memory efficient)
    # If that fails, try regular AdamW with foreach=False (uses less memory but slower)
    # As last resort, use SGD (much less memory but may need higher LR)
    optimizer = None
    try:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, fused=True)
        print("Using fused AdamW optimizer for better memory efficiency")
    except (TypeError, ValueError):
        try:
            # foreach=False uses less memory but is slower
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, foreach=False)
            print("Using standard AdamW optimizer (foreach=False for memory efficiency)")
        except Exception as e:
            # Last resort: SGD uses much less memory (no optimizer states)
            # Note: may need higher learning rate (e.g., 10x) for similar convergence
            print(f"AdamW failed with error: {e}")
            print("Falling back to SGD optimizer (uses much less memory)")
            optimizer = torch.optim.SGD(trainable_params, lr=lr * 10, momentum=0.9)
            print("WARNING: Using SGD optimizer. Consider increasing learning rate if convergence is poor.")
    
    if torch.cuda.is_available():
        print(f"GPU memory after optimizer creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
    available_mask = np.ones(n_samples, dtype=bool)

    # Clear cache before training to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for step in range(n_steps):
        chosen_indices = bandit.select_batch(batch_size, available_mask=available_mask)
        #pdb.set_trace() 
        # If you want single-use samples:
        # available_mask[chosen_indices] = False

        batch_subset = Subset(train_dataset, chosen_indices.tolist())
        loader = DataLoader(
            batch_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        batch = next(iter(loader))

        inputs = build_qwen_inputs(processor, batch, device, max_length=max_length)
        
        # Delete batch immediately to free memory (inputs contains all needed data)
        del batch, batch_subset, loader

        # loss_before - compute with no_grad and immediately move to CPU to free GPU memory
        model.eval()
        with torch.no_grad():
            losses_before, _ = compute_per_sample_loss(model, inputs)
        losses_before_cpu = losses_before.detach().cpu().numpy()
        mean_loss_before = losses_before.mean().item()  # Store mean before deleting
        del losses_before

        # Training forward pass - compute loss for gradient update
        model.train()
        
        # Ensure model is in training mode and parameters require gradients
        # This is critical for 8-bit quantized models with LoRA
        for name, param in model.named_parameters():
            if param.requires_grad:
                break
        else:
            raise RuntimeError("No parameters require gradients! Check LoRA configuration.")
        
        losses_per_sample, outputs = compute_per_sample_loss(model, inputs)
        loss_mean = losses_per_sample.mean()
        
        # Verify loss has gradients before backward pass
        if not loss_mean.requires_grad:
            raise RuntimeError(f"Loss does not require gradients! This usually means the model parameters are frozen. "
                             f"Check that LoRA is properly applied and model is in training mode.")
        
        # Delete outputs immediately to free memory before backward
        del outputs
        
        # Aggressively clear cache before backward pass (gradient checkpointing needs memory for recomputation)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        optimizer.zero_grad()
        
        # Clear cache again right before backward
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        loss_mean.backward()
        
        # Delete loss tensors before optimizer step
        del loss_mean, losses_per_sample
        
        # Aggressively clear cache before optimizer step (optimizer states are memory-intensive)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        optimizer.step()
        
        # Clear cache after optimizer step
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # loss_after - compute with model in eval mode to save memory
        model.eval()
        with torch.no_grad():
            losses_after, _ = compute_per_sample_loss(model, inputs)
        losses_after_cpu = losses_after.detach().cpu().numpy()
        mean_loss_after = losses_after.mean().item()  # Store mean before deleting
        del losses_after
        
        # Delete inputs immediately after computing loss_after
        del inputs

        per_sample_reward = (losses_before_cpu - losses_after_cpu)
        mean_reward = per_sample_reward.mean()  # Store mean before deleting
        # Use chosen_indices directly since Subset remaps indices to local (0,1,2...)
        # The order matches because shuffle=False in DataLoader
        bandit.update(chosen_indices, per_sample_reward)
        
        # Delete reward tensor
        del losses_before_cpu, losses_after_cpu, per_sample_reward
        
        # Clear cache periodically to prevent memory fragmentation
        if torch.cuda.is_available() and step % 10 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if step % 50 == 0:
            print(
                f"[step {step}] "
                f"mean_loss_before={mean_loss_before:.4f}, "
                f"mean_loss_after={mean_loss_after:.4f}, "
                f"mean_reward={mean_reward:.4f}"
            )

    return bandit, clip_embs, predictor_scores


# ------------------------------------------------
# Main
# ------------------------------------------------

def main(args):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Available CUDA devices: {device_count}")
        if args.gpu_id >= device_count:
            args.gpu_id = 0
        device = f"cuda:{args.gpu_id}"
        torch.cuda.set_device(args.gpu_id)
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    print(f"Loading Qwen2.5-VL model: {args.model_id}...")
    # Use flash_attention_2 only if CUDA is available AND flash_attn is installed
    # Otherwise fall back to sdpa (scaled dot product attention)
    if torch.cuda.is_available() and FLASH_ATTN_AVAILABLE:
        attn_impl = "flash_attention_2"
        print("Using FlashAttention2 for faster and more memory-efficient attention")
    else:
        attn_impl = "sdpa"
        if not torch.cuda.is_available():
            print("CUDA not available, using sdpa (scaled dot product attention)")
        elif not FLASH_ATTN_AVAILABLE:
            print("FlashAttention2 not available, using sdpa (scaled dot product attention)")
    
    # Use 8-bit quantization (QLoRA) if requested - reduces base model memory from ~14GB to ~7GB
    quantization_config = None
    if args.use_8bit and BITSANDBYTES_AVAILABLE and torch.cuda.is_available():
        print("Using 8-bit quantization (QLoRA) to reduce base model memory")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        # When using 8-bit, we can't specify device_map manually
        device_map_arg = "auto"
    elif args.use_8bit and not BITSANDBYTES_AVAILABLE:
        raise ImportError("8-bit quantization requested but bitsandbytes not available. Install with: pip install bitsandbytes")
    else:
        # For single GPU, use explicit device placement instead of "auto" to reduce memory overhead
        device_map_arg = device if torch.cuda.is_available() else None
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map_arg,
        low_cpu_mem_usage=True,
        use_cache=False,  # Disable cache to save memory during training
        attn_implementation=attn_impl,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    # Set padding side to 'left' for Flash Attention compatibility
    if torch.cuda.is_available() and attn_impl == "flash_attention_2":
        processor.tokenizer.padding_side = "left"
    print("Qwen2.5-VL model loaded successfully")
    
    # Apply LoRA if requested (dramatically reduces trainable parameters and optimizer memory)
    if args.use_lora and PEFT_AVAILABLE:
        print(f"Applying LoRA with rank={args.lora_r}, alpha={args.lora_alpha}, target_modules={args.lora_target_modules}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,  # LoRA rank
            lora_alpha=args.lora_alpha,  # LoRA alpha (scaling factor)
            target_modules=args.lora_target_modules.split(",") if isinstance(args.lora_target_modules, str) else args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Show how many parameters are trainable
        print("LoRA applied successfully - only LoRA parameters will be trained")
    elif args.use_lora and not PEFT_AVAILABLE:
        raise ImportError("LoRA requested but PEFT library not available. Install with: pip install peft")

    vision_embedder = CLIPEmbedder(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )

    # Load dataset / chunk
    ds = load_dataset("russwang/ThinkLite-VL-70k")
    df = ds["train"].to_pandas()
    datas = df.to_dict(orient="records")
    data_chunk = get_chunk(datas, args.num_chunks, args.chunk_idx)
    train_dataset = VQADatasetBytes(data_chunk)

    # Load predictor trained from MCTS labels
    ckpt = torch.load(args.predictor_path, map_location="cpu")
    emb_dim = ckpt["emb_dim"]
    hidden_dim = ckpt.get("hidden_dim", 512)
    predictor = SimpleMLPPredictor(emb_dim, hidden_dim=hidden_dim)
    predictor.load_state_dict(ckpt["state_dict"])
    predictor.eval()
    print(f"Loaded predictor from {args.predictor_path} with emb_dim={emb_dim}, hidden_dim={hidden_dim}")

    # Run MAB-guided training
    bandit, clip_embs, predictor_scores = train_with_mab_clip(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        clip_embedder=vision_embedder,
        predictor=predictor,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        tau=args.tau,
        tau_decay=args.tau_decay,
        device=device,
        max_length=args.max_length,
    )

    np.save("clip_embs.npy", clip_embs)
    np.save("predictor_scores.npy", predictor_scores)
    
    # Save model - if LoRA is used, save adapter weights separately
    if args.use_lora and PEFT_AVAILABLE and hasattr(model, 'save_pretrained'):
        # Save LoRA adapter weights
        model.save_pretrained(args.output_model_path.replace(".pt", "_lora"))
        print(f"LoRA adapter saved to {args.output_model_path.replace('.pt', '_lora')}")
        # Also save full state dict for compatibility
        torch.save(model.state_dict(), args.output_model_path)
    else:
        torch.save(model.state_dict(), args.output_model_path)
    print("Training finished and artifacts saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)

    # Softmax bandit hyperparams
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--tau-decay", type=float, default=0.999)

    parser.add_argument("--predictor-path", type=str, default="predictor.pt")
    parser.add_argument("--output-model-path", type=str, default="qwen_mab_posttrained.pt")
    
    # LoRA/QLoRA parameters (dramatically reduces memory usage)
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning (reduces memory by ~10-100x)")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization (QLoRA) - reduces base model memory from ~14GB to ~7GB")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (higher = more parameters, default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor (default: 32, typically 2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate (default: 0.05)")
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", 
                        help="Comma-separated list of modules to apply LoRA to (default: attention and MLP layers)")
    
    # Memory optimization: reduce sequence length to save activation memory
    parser.add_argument("--max-length", type=int, default=1024, 
                        help="Maximum sequence length (default: 1024, reduce to save memory during backward pass)")

    args = parser.parse_args()
    main(args)
