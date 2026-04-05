#!/usr/bin/env python3
"""Download OpenWebText and save as text files for prepare_data."""
import os
import sys
import multiprocessing as mp

def write_chunk(args):
    ds, start, end, out_dir, idx = args
    chunk = ds.select(range(start, end))
    path = os.path.join(out_dir, f"chunk_{idx:04d}.txt")
    # Batched column access is 100x faster than row-by-row
    texts = chunk["text"]
    with open(path, "w") as f:
        f.write("\n\n".join(texts))
    return path, end - start

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/media/volume/Prep_and_Voice_Training/data/openwebtext_raw"
    os.makedirs(out_dir, exist_ok=True)

    # Force cache to volume
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = os.path.join(os.path.dirname(out_dir), ".hf_cache")

    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets")
        sys.exit(1)

    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    print(f"Downloading OpenWebText (split={split}) to {out_dir} ...")
    ds = load_dataset("Skylion007/openwebtext", split=split, num_proc=8)
    n = len(ds)
    print(f"Got {n} documents, writing chunks...")

    chunk_size = 100000
    jobs = []
    for i in range(0, n, chunk_size):
        jobs.append((ds, i, min(i + chunk_size, n), out_dir, i // chunk_size))

    # Parallel write
    nproc = min(len(jobs), mp.cpu_count())
    with mp.Pool(nproc) as pool:
        for path, count in pool.imap_unordered(write_chunk, jobs):
            print(f"  {path} ({count} docs)")

    print("Done")

if __name__ == "__main__":
    main()
