import os
import math
import json
import io
import random
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from tqdm import tqdm
from safetensors.torch import load_file

from torch.utils.data import Dataset, DataLoader, Subset

import open_clip

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


# -----------------------------
# Helpers for chunking / IO
# -----------------------------

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
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


# -----------------------------
# Dataset: one arm per sample
# -----------------------------

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
        image_raw = d["image"]       # could be bytes or a PIL.Image
        problem = d["problem"]
        answer = d["answer"]

        # Extract question text after <image>
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


# -----------------------------
# CLIP embedder
# -----------------------------

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


# -----------------------------
# Per-sample UCB bandit
# -----------------------------

class UCBBandit:
    def __init__(self, n_arms, prior_mean=None, c=1.0):
        """
        n_arms: number of training samples
        prior_mean: np.array[n_arms], predictor-based usability scores
        c: exploration coefficient
        """
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float32)
        self.total_pulls = 0
        self.c = c

        if prior_mean is not None:
            p = prior_mean.astype(np.float32)
            p = (p - p.min()) / (p.max() - p.min() + 1e-8)
            self.values[:] = p

    def select_batch(self, k, available_mask=None):
        t = max(self.total_pulls, 1)
        ucb = self.values + self.c * np.sqrt(
            np.log(t + 1.0) / (self.counts + 1e-6)
        )

        if available_mask is not None:
            ucb = np.where(available_mask, ucb, -np.inf)

        if k >= len(ucb):
            chosen = np.argsort(-ucb)
        else:
            chosen = np.argpartition(-ucb, k)[:k]
        return chosen

    def update(self, chosen_indices, reward):
        """
        chosen_indices: np.array of global indices
        reward: scalar OR array of shape (len(chosen_indices),)
        """
        self.total_pulls += len(chosen_indices)

        if np.isscalar(reward):
            rewards = np.full(len(chosen_indices), reward, dtype=np.float32)
        else:
            rewards = np.asarray(reward, dtype=np.float32)

        for idx, r in zip(chosen_indices, rewards):
            self.counts[idx] += 1
            n = self.counts[idx]
            self.values[idx] += (r - self.values[idx]) / n


# -----------------------------
# Predictor on CLIP embeddings
# -----------------------------

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


# -----------------------------
# Qwen2.5-VL input & loss
# -----------------------------

def build_qwen_inputs(processor, batch, device):
    images = []
    for img_raw in batch["images"]:
        if isinstance(img_raw, Image.Image):
            img = img_raw
        else:
            img = Image.open(io.BytesIO(img_raw)).convert("RGB")
        images.append(img)

    questions = batch["questions"]
    answers = batch["answers"]

    prompts = [f"Question: {q}\nAnswer:" for q in questions]
    texts = [p + " " + a for p, a in zip(prompts, answers)]

    proc_inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    input_ids = proc_inputs["input_ids"]
    labels = input_ids.clone()

    tokenizer = processor.tokenizer
    for i, p in enumerate(prompts):
        tok_prompt = tokenizer(p, return_tensors="pt")
        prompt_len = tok_prompt.input_ids.size(1)
        labels[i, :prompt_len] = -100  # ignore prompt

    proc_inputs["labels"] = labels
    return proc_inputs


def compute_per_sample_loss(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits  # (B, T, V)
    labels = inputs["labels"]  # (B, T)

    vocab_size = logits.size(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    )  # (B*(T-1),)

    loss_flat = loss_flat.view(shift_labels.size(0), -1)  # (B, T-1)

    mask = (shift_labels != -100).float()
    loss_flat = loss_flat * mask

    losses_per_sample = loss_flat.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B,)
    return losses_per_sample, outputs


# -----------------------------
# MAB-guided training loop
# -----------------------------

def train_with_mab_clip(
    model,
    processor,
    train_dataset,      # VQADatasetBytes
    clip_embedder,
    predictor,
    batch_size=8,
    n_steps=1000,
    lr=1e-5,
    c_ucb=1.0,
    device="cuda",
):
    model.to(device)
    model.train()

    # 1) CLIP embeddings + predictor scores
    clip_embs, predictor_scores = compute_clip_embeddings_and_predictor_scores(
        dataset=train_dataset,
        clip_embedder=clip_embedder,
        predictor=predictor,
        batch_size=batch_size,
        device=device,
    )

    n_samples = len(train_dataset)
    bandit = UCBBandit(n_arms=n_samples, prior_mean=predictor_scores, c=c_ucb)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    available_mask = np.ones(n_samples, dtype=bool)

    for step in range(n_steps):
        # 2) choose per-sample arms
        chosen_indices = bandit.select_batch(batch_size, available_mask=available_mask)
        # If you want single-use samples, uncomment:
        # available_mask[chosen_indices] = False

        batch_subset = Subset(train_dataset, chosen_indices.tolist())
        loader = DataLoader(
            batch_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        batch = next(iter(loader))
        global_idxs = batch["idxs"].cpu().numpy()

        # 3) build inputs
        inputs = build_qwen_inputs(processor, batch, device)

        # 4) loss_before
        losses_before, _ = compute_per_sample_loss(model, inputs)
        loss_mean = losses_before.mean()

        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

        # 5) loss_after
        with torch.no_grad():
            losses_after, _ = compute_per_sample_loss(model, inputs)

        # 6) reward per sample = loss_before - loss_after
        per_sample_reward = (losses_before - losses_after).detach().cpu().numpy()

        # 7) update bandit on GLOBAL indices
        bandit.update(global_idxs, per_sample_reward)

        if step % 50 == 0:
            print(
                f"[step {step}] "
                f"mean_loss_before={losses_before.mean().item():.4f}, "
                f"mean_loss_after={losses_after.mean().item():.4f}, "
                f"mean_reward={per_sample_reward.mean():.4f}"
            )

    return bandit, clip_embs, predictor_scores


# -----------------------------
# Main
# -----------------------------

def main(args):
    # Device
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

    # Load Qwen2.5-VL model
    print(f"Loading Qwen2.5-VL model: {args.model_id}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        use_cache=True,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    print("Qwen2.5-VL model loaded successfully")

    # CLIP embedder (required now)
    vision_embedder = CLIPEmbedder(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )

    # Load dataset and chunk
    ds = load_dataset("russwang/ThinkLite-VL-70k")
    df = ds["train"].to_pandas()
    datas = df.to_dict(orient="records")
    data_chunk = get_chunk(datas, args.num_chunks, args.chunk_idx)

    # Build dataset
    train_dataset = VQADatasetBytes(data_chunk)

    # Build predictor (you can replace with your MCTS-trained model + load weights)
    sample_emb = vision_embedder.embed(
        data_chunk[0]["image"],
        data_chunk[0]["problem"].split("<image>")[1],
    )
    emb_dim = sample_emb.shape[-1]
    predictor = SimpleMLPPredictor(emb_dim)
    # predictor.load_state_dict(torch.load("your_predictor.pt"))

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
        c_ucb=args.exploration_c,
        device=device,
    )

    # Save embeddings / scores if desired
    np.save("clip_embs.npy", clip_embs)
    np.save("predictor_scores.npy", predictor_scores)
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
    parser.add_argument("--exploration-c", type=float, default=2.0)

    parser.add_argument("--output-model-path", type=str, default="qwen_mab_posttrained.pt")

    args = parser.parse_args()
    main(args)
