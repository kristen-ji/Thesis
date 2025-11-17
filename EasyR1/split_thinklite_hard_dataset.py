#!/usr/bin/env python3
"""
Split ThinkLite-VL-hard-11k dataset into train and validation sets.
Usage: python split_thinklite_hard_dataset.py --train_ratio 0.8 --output_dir ./
"""
import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

def _materialize_images(df: pd.DataFrame, image_dir: str, split_name: str) -> pd.DataFrame:
    """
    Save image bytes from the dataframe to local files and create an `images` column
    with relative file names expected by EasyR1 (list[str] per row).

    Supported input columns (first found is used): `image`, `image_data`.
    """
    os.makedirs(image_dir, exist_ok=True)

    source_col = None
    for candidate in ["image", "image_data"]:
        if candidate in df.columns:
            source_col = candidate
            break

    if source_col is None:
        raise ValueError("No image bytes column found. Expected one of: 'image', 'image_data'.")

    # Prepare output column
    image_filenames = []

    for idx, row in df.iterrows():
        data = row[source_col]
        # data can be raw bytes, dict with 'bytes', or a list/array
        if isinstance(data, (bytes, bytearray)):
            img_bytes = data
        elif isinstance(data, dict) and "bytes" in data:
            img_bytes = data["bytes"]
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            # Assume first image per sample
            first = data[0]
            if isinstance(first, (bytes, bytearray)):
                img_bytes = first
            elif isinstance(first, dict) and "bytes" in first:
                img_bytes = first["bytes"]
            else:
                raise ValueError("Unsupported image element type in list for row %d" % idx)
        else:
            raise ValueError("Unsupported image field type for row %d" % idx)

        filename = f"{split_name}_{idx:07d}.png"
        out_path = os.path.join(image_dir, filename)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        image_filenames.append([filename])  # EasyR1 expects a list[str]

    df = df.copy()
    df["images"] = image_filenames
    # Drop the original bytes column to avoid accidental text-only path
    df = df.drop(columns=[source_col])
    return df


def split_dataset(train_ratio=0.8, output_dir="./", random_seed=42, shuffle=True,
                  materialize_images=False, image_dir: str = "") -> Tuple[str, str]:
    """
    Split ThinkLite-VL-hard-11k dataset into train and validation sets.
    
    Args:
        train_ratio: Ratio of training data (e.g., 0.8 means 80% train, 20% val)
        output_dir: Directory to save the output parquet files
        random_seed: Random seed for reproducibility
        shuffle: Whether to shuffle the dataset before splitting
    """
    print("Loading ThinkLite-VL-hard-11k dataset from Hugging Face...")
    ds = load_dataset("russwang/ThinkLite-VL-hard-11k")
    
    # Convert to pandas DataFrame
    df = ds['train'].to_pandas()
    total_samples = len(df)
    print(f"Total samples: {total_samples:,}")
    
    # Calculate split sizes
    train_size = int(total_samples * train_ratio)
    val_size = total_samples - train_size
    print(f"\nSplit ratio: {train_ratio:.1%} train, {(1-train_ratio):.1%} validation")
    print(f"Train samples: {train_size:,}")
    print(f"Val samples: {val_size:,}")
    
    # Split the dataset
    if shuffle:
        # Shuffle using numpy
        np.random.seed(random_seed)
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        print(f"\nDataset shuffled with random_seed={random_seed}")
    else:
        # Sequential split (first N rows for train, rest for val)
        train_df = df[:train_size].reset_index(drop=True)
        val_df = df[train_size:].reset_index(drop=True)
        print("\nSequential split (no shuffle)")
    
    # Optionally save images to local dir and produce `images` column
    if materialize_images:
        if not image_dir:
            raise ValueError("--image_dir must be provided when --materialize_images is set")
        print(f"\nMaterializing images to: {image_dir}")
        # Save images directly under image_dir with split-prefixed filenames
        train_df = _materialize_images(train_df, image_dir=image_dir, split_name="train")
        val_df = _materialize_images(val_df, image_dir=image_dir, split_name="val")

    # Save to parquet files (with `images` column if materialized)
    train_output = f"{output_dir}/thinklite_hard_train.parquet"
    val_output = f"{output_dir}/thinklite_hard_val.parquet"
    
    print(f"\nSaving training data to: {train_output}")
    train_df.to_parquet(train_output, compression='snappy', index=False)
    
    print(f"Saving validation data to: {val_output}")
    val_df.to_parquet(val_output, compression='snappy', index=False)
    
    # Show file sizes
    import os
    train_size_mb = os.path.getsize(train_output) / 1024**2
    val_size_mb = os.path.getsize(val_output) / 1024**2
    
    print(f"\n{'='*60}")
    print("✓ Dataset split completed!")
    print(f"{'='*60}")
    print(f"Train file: {train_output} ({train_size_mb:.1f} MB)")
    print(f"Val file:   {val_output} ({val_size_mb:.1f} MB)")
    print(f"\nYou can now use these files in your training script:")
    print(f"  data.train_files={train_output}")
    print(f"  data.val_files={val_output}")
    if materialize_images:
        print(f"  data.image_dir={image_dir}")

    return train_output, val_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split ThinkLite-VL-hard-11k dataset into train and validation sets'
    )
    parser.add_argument(
        '--train_ratio', 
        type=float, 
        default=0.8, 
        help='Ratio of training data (default: 0.8 for 80%% train, 20%% val)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./', 
        help='Directory to save output parquet files (default: current directory)'
    )
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=42, 
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no_shuffle', 
        action='store_true', 
        help='Disable shuffling (use sequential split instead)'
    )
    parser.add_argument(
        '--materialize_images',
        action='store_true',
        help='Save images to local files and create an images column with relative paths'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Directory to save images when --materialize_images is set'
    )
    
    args = parser.parse_args()
    
    split_dataset(
        train_ratio=args.train_ratio,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        shuffle=not args.no_shuffle,
        materialize_images=args.materialize_images,
        image_dir=args.image_dir,
    )

