# Mac local bench — 125M / fast-inference (post all 23 optimizations)

## Setup
- Hardware: this Mac (Apple Silicon), MPS backend
- Model: `checkpoints/125M.pt` — 12 layers, d_model=768, n_heads=12, vocab=50257
- Workload: prompt_len=128, decode_len=64, batch=1, greedy, fp32
- Iters: 3 timed + 2 warmup

## Results

| Stack | TTFT (ms) | TPOT mean (ms) | TPOT p99 (ms) | tok/s | Note |
|---|---|---|---|---|---|
| **llm-cpp** (our C++, fast-inference branch with all 23 opts) | 17.1 | 5.34 | 7.26 | **187.4** | bench_chat on real 125M weights |
| **Vanilla PyTorch (eager, nn.TransformerEncoder)** | 31.8 | 40.1 | 46.5 | 24.9 | bench_pytorch_125M.py fallback (no olmo-core) |

**llm-cpp is 7.5× faster than vanilla PyTorch eager mode on the same Mac MPS at the same parameter scale.**

## Caveats (honest)

- The PyTorch baseline used the `nn.TransformerEncoder` *fallback path* because `olmo_core` isn't importable from `.bench_venv` despite the `.pth` file. nn.TransformerEncoder is post-LN-flavored MHA — typically slightly slower than OLMo-2's pre-LN+SwiGLU+RoPE+QK-norm. Number is directional, not a head-to-head architectural match.
- llm-cpp ran the real 125M trained weights; the PyTorch baseline ran random-init at the same shape (it's measuring kernel speed, not loss). Fair for tok/s comparison.
- **Most of our 23 optimizations don't fire on this Mac** — they live in CUDA-only paths (warp-shuffles, float4 vec, BF16 paired loads, fused dequant). On Mac the wins come from:
  - The cumulative-sum + binary search sampler (chat.cpp) instead of `discrete_distribution`
  - The sliding-window mask kernel reduction
  - The eager LM-head + sampler structure already in fast-inference
  - LibTorch's own ATen optimizations on MPS
- The pre-optimization baseline of fast-inference itself **does not compile** on macOS clang (pre-existing const-correctness bug in `lm_head.hpp` that I fixed in round 1). So we can't produce a clean "before-our-23-changes" number for this Mac without first applying that fix. Real "before vs after our changes" measurement happens on H100.

## bench_attn (kernel-isolated) curves on this Mac

### CPU prefill — quadratic vs linear-ish

| T | dense (ms) | sparse (ms) | speedup |
|---|---|---|---|
| 1024 | 7.0 | 29.9 | 0.23× (sparse overhead dominates) |
| 4096 | 73.7 | 199.1 | 0.37× |
| 16384 | 1165.0 | 892.5 | 1.31× |
| 32768 | 4737.1 | 1710.9 | **2.77×** |

Sparse prefill catches dense around T=8K and grows the lead from there.

### MPS decode (KV cache T, single-query) — sparse is dominant at long context

| T | dense (ms) | sparse (ms) | speedup |
|---|---|---|---|
| 1024 | 0.94 | 1.76 | 0.53× |
| 4096 | 3.55 | 1.44 | 2.47× |
| 16384 | 14.13 | 1.63 | **8.66×** |

Sparse decode stays nearly flat as T grows (k_top × block_size = 1024 attended positions regardless of T). Dense grows linearly. **This is the SubQ effect, reproduced on this Mac.**

## Files
- `results/bench_chat_olmocpp_mps.json` — our C++ end-to-end
- `results/bench_pytorch_mps.json` — vanilla PyTorch baseline
- `/tmp/subq_mac_dense_pre.csv`, etc. — kernel-isolated CSVs
