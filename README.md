# cpp-llm

Pure C++ implementation of the OLMo-2 transformer architecture on LibTorch. Train, evaluate, and chat with language models — no Python required at runtime.

## Features

- **Full OLMo-2 architecture**: RMSNorm, RoPE, GQA, SwiGLU FFN, MoE
- **Training**: AdamW with cosine warmup LR, gradient accumulation, gradient clipping, DDP
- **Data pipeline**: GPT-2 BPE tokenizer, `.npy` token datasets, variable-length batching
- **Inference**: Interactive chat CLI with KV cache, top-k/top-p sampling, repetition penalty
- **Checkpointing**: Save/load `.pt` checkpoints, HuggingFace safetensors import
- **Distributed**: DDP support (requires Gloo), activation checkpointing, mixed precision
- **Evaluation**: Perplexity, LM evaluation harness integration
- **Configs**: Presets for 33M, 125M, 3B, 7B models

## Prerequisites

- CMake >= 3.18
- C++17 compiler (clang or gcc)
- LibTorch (download from https://pytorch.org)
- nlohmann/json (auto-fetched if not found)
- zlib

## Build

```bash
# Set LibTorch path
export LIBTORCH_PATH=/path/to/libtorch

# Configure and build
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH
make -j$(nproc)
```

This produces:
- `olmo_train` — Training executable
- `chat` — Interactive chat CLI
- `prepare_data` — BPE tokenization tool
- `convert_hf` — HuggingFace to .pt converter
- `convert_checkpoint` — Checkpoint format converter
- `dump_params` — Debug parameter dumper

## Quick Start

### 1. Prepare data

```bash
# Download GPT-2 tokenizer files
mkdir -p data/gpt2
curl -L https://huggingface.co/gpt2/resolve/main/vocab.json -o data/gpt2/vocab.json
curl -L https://huggingface.co/gpt2/resolve/main/merges.txt -o data/gpt2/merges.txt

# Tokenize your text data
./build/prepare_data \
  --input data/stories.jsonl \
  --output data/stories.npy \
  --vocab data/gpt2/vocab.json \
  --merges data/gpt2/merges.txt \
  --max-stories 500000
```

### 2. Train a model

```bash
./build/olmo_train --train \
  --config configs/olmo2_33M_fast.json \
  --data-path data/stories.npy \
  --batch-size 8 --seq-len 256 \
  --steps 5000 --lr 3e-4 --warmup-steps 200 \
  --device mps \
  --save checkpoints/33M.pt
```

### 3. Chat with your model

```bash
./build/chat \
  --checkpoint checkpoints/33M.pt \
  --config configs/olmo2_33M_fast.json \
  --vocab-file data/gpt2/vocab.json \
  --merges-file data/gpt2/merges.txt \
  --device mps
```

### 4. Load a HuggingFace OLMo-2 checkpoint

```bash
# Download OLMo-2-7B from HuggingFace, then:
./build/convert_hf \
  --hf-dir /path/to/OLMo-2-7B \
  --config configs/olmo2_7B.json \
  --output checkpoints/olmo2_7B.pt
```

## Model Configs

| Config | d_model | Layers | Heads | Params |
|--------|---------|--------|-------|--------|
| `olmo2_33M_fast.json` | 512 | 8 | 8 | 33M |
| `olmo2_125M.json` | 768 | 12 | 12 | 125M |
| `olmo2_3B.json` | 2560 | 32 | 20 | 3B |
| `olmo2_7B.json` | 4096 | 32 | 32 | 7B |

## Architecture

```
Embedding → RMSNorm → [TransformerBlock × N] → LMHead(RMSNorm → Linear)

TransformerBlock:
  h = x + RMSNorm(Attention(x))
  out = h + RMSNorm(FeedForward(h))

Attention: Multi-head with RoPE, GQA support, QK norm
FeedForward: SwiGLU (gate * silu(up)) with down projection
```

## Project Structure

```
include/olmo_cpp/     # Headers
  config.hpp          # TransformerConfig
  model/              # Transformer, Attention, FFN, RoPE, LMHead, MoE
  data/               # BPE tokenizer, token dataset, collator
  optim/              # LR scheduler
  train/              # Callbacks, checkpointing, grad scaler
  eval/               # Evaluator, metrics
  distributed/        # DDP, tensor/pipeline/expert parallelism
src/                  # Implementation
  main.cpp            # Training CLI
  train.cpp           # Training loop
tools/                # Standalone executables
  chat.cpp            # Interactive chat
  prepare_data.cpp    # Data tokenization
  convert_hf.cpp      # HF checkpoint import
configs/              # Model configuration JSONs
```
