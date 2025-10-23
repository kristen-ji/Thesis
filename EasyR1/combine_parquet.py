#!/usr/bin/env python3
"""
Combine multiple parquet files into one.
Usage: python combine_parquet.py file1.parquet file2.parquet ... -o output.parquet
"""
import sys
import pandas as pd
import argparse

def combine_parquet_files(input_files, output_file):
    """Combine multiple parquet files into one."""
    print(f"Combining {len(input_files)} parquet files...")
    
    dfs = []
    for i, file in enumerate(input_files, 1):
        print(f"  [{i}/{len(input_files)}] Reading {file}...")
        df = pd.read_parquet(file)
        print(f"    Rows: {len(df):,}")
        dfs.append(df)
    
    print("\nConcatenating dataframes...")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(combined):,}")
    
    print(f"\nSaving to {output_file}...")
    combined.to_parquet(output_file, compression='snappy', index=False)
    
    # Show size
    import os
    size_mb = os.path.getsize(output_file) / 1024**2
    print(f"Done! Output size: {size_mb:.1f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine multiple parquet files')
    parser.add_argument('input_files', nargs='+', help='Input parquet files')
    parser.add_argument('-o', '--output', required=True, help='Output parquet file')
    
    args = parser.parse_args()
    combine_parquet_files(args.input_files, args.output)
