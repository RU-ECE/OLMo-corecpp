#!/usr/bin/env python3
"""Download TinyStories dataset from HuggingFace and save as text files.

Requires: pip install datasets
Usage: python3 scripts/download_tinystories.py [--max-stories N]
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stories", type=int, default=500000,
                        help="Max stories to download (default: 500k)")
    parser.add_argument("--output-dir", default="data/tinystories_raw",
                        help="Output directory for text files")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets --quiet")
        from datasets import load_dataset

    print(f"Downloading TinyStories (up to {args.max_stories} stories)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    os.makedirs(args.output_dir, exist_ok=True)

    # Write stories to text files (batched for efficiency)
    batch_size = 10000
    total = min(len(ds), args.max_stories)
    file_idx = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        fpath = os.path.join(args.output_dir, f"stories_{file_idx:04d}.txt")
        with open(fpath, "w") as f:
            for i in range(start, end):
                text = ds[i]["text"].strip()
                if text:
                    f.write(text + "\n\n")
        file_idx += 1
        print(f"\r  Saved {end}/{total} stories...", end="", flush=True)

    print(f"\nDone! {total} stories saved to {args.output_dir}/")
    print(f"Files: {file_idx} files")

if __name__ == "__main__":
    main()
