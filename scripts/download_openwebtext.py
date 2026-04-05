#!/usr/bin/env python3
"""Download OpenWebText and save as text files for prepare_data."""
import os
import sys

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/media/volume/Prep_and_Voice_Training/data/openwebtext_raw"
    os.makedirs(out_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets")
        sys.exit(1)

    print(f"Downloading OpenWebText to {out_dir} ...")
    ds = load_dataset("Skylion007/openwebtext", split="train", num_proc=8)
    print(f"Got {len(ds)} documents")

    chunk_size = 100000
    for i in range(0, len(ds), chunk_size):
        end = min(i + chunk_size, len(ds))
        chunk = ds.select(range(i, end))
        path = os.path.join(out_dir, f"chunk_{i // chunk_size:04d}.txt")
        with open(path, "w") as f:
            for row in chunk:
                f.write(row["text"] + "\n\n")
        print(f"  Wrote {path} ({end - i} docs)")

    print("Done")

if __name__ == "__main__":
    main()
