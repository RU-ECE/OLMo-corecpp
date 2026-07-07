# 1B Inference Benchmark — Apple M4 Pro (MPS)

Single-stream and batched decode of the trained **1B** model (973M params, d2048/16L/16H,
GPT-2 vocab) on an Apple M4 Pro via the MPS backend, versus ollama at matched 1B scale.

- **Date:** 2026-07-03
- **Hardware:** Apple M4 Pro, unified memory, Metal Performance Shaders (MPS) backend
- **Model:** `runs/kuiper_1B/model.pt` (bf16 checkpoint, 1.81 GiB; pulled from Box `SatrajitShared/Jetstream stuff/llmcpp_preservation/runs_1B/model.pt`)
- **Config:** `configs/kuiper_1B.json` · GPT-2 BPE tokenizer
- **Shape:** prompt_len 128, decode_len 256, warmup 1, iters 3
- **Engines:** our C++ engine (`bench_chat`) vs ollama `llama3.2:1b` (Q4, Metal)

## Results

| Engine / config | batch 1 | batch 8 | batch 32 |
|---|---|---|---|
| **Our C++ engine (bf16)** | **91.8 tok/s** | **345.1 tok/s** | **426.2 tok/s** |
| ollama `llama3.2:1b` (Q4) | 152.8 tok/s | — | — |

**Latency detail (our C++ engine):**

| batch | TTFT (prefill) | TPOT (per output token) | Throughput |
|---|---|---|---|
| 1  | 55.6 ms   | 10.9 ms (p50 10.9, p99 11.9) | 91.8 tok/s |
| 8  | 381.9 ms  | 23.2 ms (p50 23.2, p99 29.6) | 345.1 tok/s |
| 32 | 1520.2 ms | 75.1 ms (p50 74.4, p99 103.2) | 426.2 tok/s |

## How to read it

- **Single-stream (batch 1): ollama is faster — 152.8 vs 91.8 tok/s — and this is expected, not a fair fight.**
  ollama runs **4-bit Q4**; our row is **16-bit bf16** (≈2× the memory traffic per token), and Apple's
  Metal backend is highly mature. The apples-to-apples 4-bit comparison (our INT4 vs ollama Q4) does not
  exist on this platform — see "INT4" below.
- **Batched throughput is the C++ engine's edge:** **426 tok/s at batch 32 ≈ 2.8× ollama's single-stream.**
  A batched server saturates the GPU; single-stream ollama cannot match aggregate tok/s. This is the
  intended use case for the engine (serving many concurrent requests), and the reason the batch-8/32
  rows matter more than batch-1 for the systems argument.

## INT4 — not available on MPS (CUDA-only)

INT4 could not be benchmarked on the Mac, for two independent reasons:

1. **Quantization step fails to load the checkpoint.** `quantize_int4` calls `torch::load(model, path)`
   without a target device; the `.pt` was serialized from a CUDA model, so deserialization tries to
   place tensors on the CUDA backend, which does not exist on the Mac
   (`Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend`). `bench_chat` avoids
   this by loading onto the target device first; `quantize_int4` does not.
2. **The INT4 decode kernel is CUDA-only.** Even with a valid `.int4.pt`, single-stream INT4 decode
   dispatches to the `int4_gemv` CUDA kernel (`src/nn/quant.cpp` / `kernels/quant_dequant.cu`), which
   has no MPS implementation.

**INT4 is a CUDA-GPU story** (RTX 4090 / H100). On the 4090 the fine-tuned/quantized path runs
INT4-vs-Q4 head-to-head; on Apple Silicon the engine tops out at **bf16 / fp32**.

## Reproduce

```bash
cd ~/Projects/llm-cpp
# model already at runs/kuiper_1B/model.pt (else fetch from Box, see header)
CKPT=runs/kuiper_1B/model.pt CONFIG=configs/kuiper_1B.json DEVICE=mps \
  bash scripts/bench/bench_1b_infer.sh
```

Report **steady-state** tok/s (the trainer/bench prints per-iter means, not cumulative averages).
For a same-precision picture, add an fp32 row (drop `--bf16` in the bf16 invocation) — MPS fp32 for
this 1B was previously verified at ~55 tok/s single-stream on this class of machine.

## Comparison context (other platforms)

| Platform | Config | Single-stream | Notes |
|---|---|---|---|
| **M4 Pro (MPS)** | 1B bf16 | 91.8 tok/s | this run; 426 tok/s @ batch 32 |
| M4 Pro (MPS) | 1B fp32 | ~55 tok/s | prior verified (`PROFESSOR_INFERENCE.md`) |
| RTX 4090 / H100 | 1B bf16 / INT4 | run on that box | INT4 (CUDA-only) is the fair vs-Q4 row |
| ollama (M4 Pro, Metal) | llama3.2:1b Q4 | 152.8 tok/s | 4-bit baseline at 1B scale |
