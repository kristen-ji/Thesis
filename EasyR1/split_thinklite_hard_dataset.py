#!/usr/bin/env python3
"""
Split ThinkLite-VL-hard-11k dataset into train and validation sets.
Usage: python split_thinklite_hard_dataset.py --train_ratio 0.8 --output_dir ./
"""
import argparse
import pandas as pd
from datasets import load_dataset
import numpy as np

def split_dataset(train_ratio=0.8, output_dir="./", random_seed=42, shuffle=True):
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
    
    # Save to parquet files
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
    
    args = parser.parse_args()
    
    split_dataset(
        train_ratio=args.train_ratio,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        shuffle=not args.no_shuffle
    )

