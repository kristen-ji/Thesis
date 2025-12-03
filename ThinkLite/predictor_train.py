import os
import io
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import open_clip


# ------------------------------------------------
# CLIP embedder (same as in mab.py)
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
        Returns: combined normalized [D] tensor on self.device
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

        return combined.squeeze(0)  # (D,)


# ------------------------------------------------
# Predictor (must match mab.py)
# ------------------------------------------------

class SimpleMLPPredictor(nn.Module):
    def __init__(self, emb_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------
# Dataset wrapping precomputed (embedding, label)
# ------------------------------------------------

class LabeledEmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        embeddings: [N, D]
        labels: [N] in {0,1}
        """
        assert embeddings.size(0) == labels.size(0)
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ------------------------------------------------
# Load MCTS parquet and extract records
# ------------------------------------------------

def load_mcts_records(parquet_paths):
    if isinstance(parquet_paths, str):
        parquet_paths = [parquet_paths]

    dfs = []
    for p in parquet_paths:
        print(f"Loading parquet: {p}")
        df = pd.read_parquet(p, engine="pyarrow")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df_all)}")

    # Expect at least: image, problem, ground_truth / answer, solution, iters
    records = df_all.to_dict(orient="records")

    filtered = []
    for d in records:
        if "image" not in d or "problem" not in d or "solution" not in d:
            continue
        if "ground_truth" not in d and "answer" not in d:
            continue
        filtered.append(d)

    print(f"Records with required fields: {len(filtered)}")
    return filtered


def compute_embeddings_and_labels(records, clip_embedder, device, min_iters_positive=5):
    """
    For each record:
      - embed (image, question)
      - label = 1 if:
          * MCTS solution is correct (solution contains ground_truth), AND
          * iters >= min_iters_positive
        else 0.
    Returns:
      embeddings: [N, D] tensor
      labels: [N] tensor in {0,1}
    """
    embs = []
    labels = []

    for d in tqdm(records, desc="Embedding + labeling"):
        img_raw = d["image"]
        problem = d["problem"]
        solution = str(d.get("solution", "")).lower()

        # extract question
        if "<image>" in problem:
            question = problem.split("<image>")[1]
        else:
            question = problem

        gt = str(d.get("ground_truth", d.get("answer", ""))).strip().lower()
        solution = str(d.get("solution", "")).lower()
        iters = d.get("iters", 0)

        # difficulty condition: requires at least min_iters_positive MCTS iterations
        is_difficult = (iters is not None) and (iters >= min_iters_positive)

        # final label: hard sample, regardless of solved/unsolved
        label = 1.0 if is_difficult else 0.0


        emb = clip_embedder.embed(img_raw, question)  # (D,)
        embs.append(emb.cpu())
        labels.append(label)

    embeddings = torch.stack(embs, dim=0).to(device)              # [N, D]
    labels = torch.tensor(labels, dtype=torch.float32).to(device) # [N]

    pos = labels.sum().item()
    neg = len(labels) - pos
    print(
        f"Embeddings shape: {embeddings.shape}, "
        f"labels: pos={pos:.0f}, neg={neg:.0f}, "
        f"min_iters_positive={min_iters_positive}"
    )
    return embeddings, labels


# ------------------------------------------------
# Training loop
# ------------------------------------------------

def train_predictor(
    embeddings,
    labels,
    emb_dim,
    hidden_dim=512,
    batch_size=64,
    lr=1e-4,
    n_epochs=10,
    weight_decay=0.0,
    device="cuda",
):
    model = SimpleMLPPredictor(emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)

    dataset = LabeledEmbeddingDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()  # predictor outputs probabilities
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(-1)  # [B] -> [B,1]

            preds = model(x)  # [B,1] in [0,1]
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(dataset)
        print(f"[Epoch {epoch+1}/{n_epochs}] loss={epoch_loss:.4f}")

    return model


# ------------------------------------------------
# Main
# ------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load MCTS parquet(s)
    records = load_mcts_records(args.parquet_paths)

    # 2) CLIP embedder (must match mab.py)
    clip_embedder = CLIPEmbedder(
        model_name=args.clip_model_name,
        pretrained=args.clip_pretrained,
    )

    # 3) Compute embeddings + labels (with difficulty condition)
    embeddings, labels = compute_embeddings_and_labels(
        records,
        clip_embedder=clip_embedder,
        device=device,
        min_iters_positive=args.min_iters_positive,
    )
    emb_dim = embeddings.size(1)

    # 4) Train predictor
    predictor = train_predictor(
        embeddings=embeddings,
        labels=labels,
        emb_dim=emb_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        n_epochs=args.epochs,
        weight_decay=args.weight_decay,
        device=device,
    )

    # 5) Save predictor in the format mab.py expects
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": predictor.state_dict(),
            "emb_dim": emb_dim,
            "hidden_dim": args.hidden_dim,
        },
        args.output_path,
    )
    print(f"Saved predictor to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # one or more parquet paths from mcts.py
    parser.add_argument(
        "--parquet-paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more parquet files output by mcts.py (must contain image, problem, ground_truth/answer, solution, iters).",
    )

    # CLIP options (must match mab.py)
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default="ViT-B-32",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="laion2b_s34b_b79k",
    )

    # Difficulty threshold for positive labels
    parser.add_argument(
        "--min-iters-positive",
        type=int,
        default=5,
        help="Minimum MCTS iterations required (in addition to correctness) to label a sample as positive.",
    )

    # Predictor training hyperparams
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument(
        "--output-path",
        type=str,
        default="predictor.pt",
        help="Where to save the trained predictor checkpoint.",
    )

    args = parser.parse_args()
    main(args)
