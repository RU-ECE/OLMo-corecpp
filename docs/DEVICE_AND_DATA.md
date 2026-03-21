# Device Support & Data Pipeline

## Why No LibTorch for Data Preparation?

The `prepare_data` tool is **intentionally standalone** (no LibTorch):

1. **No GPU needed** – Tokenization is CPU-bound: file I/O, string splitting, BPE merges. GPUs don't help.
2. **Faster builds** – prepare_data builds in seconds without pulling in LibTorch (large dependency).
3. **Smaller binary** – The tool is ~1MB vs ~500MB+ with LibTorch.
4. **Simpler deployment** – Run data prep on any machine without CUDA/Metal.

LibTorch is used only for **training** (model forward/backward, optimizer), where GPU acceleration matters.

## Metal (MPS) on Apple Silicon

Training uses Metal when available on Mac:

```bash
# Auto-detect (MPS on Mac, CUDA on Linux, else CPU)
./build/olmo_train --train --data-path data/tokens.npy --config configs/olmo2_3B.json

# Force Metal
./build/olmo_train --train --device mps --data-path data/tokens.npy

# Force CPU
./build/olmo_train --train --device cpu --data-path data/tokens.npy
```

**Requirements:**
- macOS 12.3+
- Apple Silicon (M1/M2/M3/M4)
- LibTorch built with MPS (default for `pip install torch` on Mac)

## Data for 3B Model Training

A 3B model typically needs **10B+ tokens** for meaningful training. Options:

### 1. TinyStories (quick start, ~500M tokens)

```bash
# Download GPT-2 tokenizer (one-time)
mkdir -p data/gpt2
curl -sL https://huggingface.co/gpt2/resolve/main/vocab.json -o data/gpt2/vocab.json
curl -sL https://huggingface.co/gpt2/resolve/main/merges.txt -o data/gpt2/merges.txt

# Download TinyStories from HuggingFace (100K rows ≈ 50M tokens)
./build/prepare_data --download-hf roneneldan/TinyStories --output data/tinystories.npy \
  --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt --max-tokens 50000000
```

### 2. Full TinyStories (~2M rows, ~500M tokens)

Increase `max_rows` in the code or run multiple `--download-hf` with different offsets. For full 2M rows, use the HuggingFace CLI to download Parquet, then convert:

```bash
# If you have huggingface-cli (Python)
huggingface-cli download roneneldan/TinyStories --repo-type dataset --local-dir data/tinystories_raw
# Then convert Parquet to text and run prepare_data --input
```

### 3. Larger datasets (OpenWebText, C4, The Pile)

Use `--download <url>` with a direct URL to raw text or JSONL, or download manually and use `--input`.

## 3B Training Example

```bash
# 1. Prepare data
./build/prepare_data --download-hf roneneldan/TinyStories --output data/tokens.npy \
  --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt \
  --max-tokens 100000000 --threads 8

# 2. Train on Metal (Apple Silicon)
./build/olmo_train --train --device mps --data-path data/tokens.npy \
  --config configs/olmo2_3B.json
```

**Note:** 3B on Metal needs ~16–24GB unified memory. Reduce batch_size if OOM.
