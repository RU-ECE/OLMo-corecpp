# Speed results ledger (branch `speed/phase0`)

Every measurement behind the speed plan, with how it was taken, so a paper can be written from this file.
Plan doc: https://claude.ai/artifact/XVZFJqNuN5APh8hLGwi1t6 . Raw logs: `runs/killtest/`, `runs/colab/`.

Hardware: **M4 Pro** = Apple M4 Pro, 16-core GPU, 24 GB, LibTorch 2.14 (Homebrew), MPS, fp32.
**A100** = Colab Pro, A100-SXM4-40GB (sm_80), torch 2.14.1+cu130, CUDA 13.0.
All timings are medians of repeated runs after warm-up.

## 1. Correctness fixes (each test fails on the old code, passes now)

| Bug | Fix | Proof (old → new) | Test |
|---|---|---|---|
| 1 cuBLASLt `fast_linear` outside autograd | fast path only when autograd is not recording | A100: 12/30 params got a gradient → 30/30 | `test_grad_coverage cuda` |
| 2 DDP hooks | post-accumulate hooks reduce `.grad`; sync only the last micro-batch; double reduce is an error | 2 ranks, 3 steps: old ranks 0.22 apart even at accumulation 1; new = reference to 1e-16 | `test_ddp_accum` |
| 3 zwt `fire_idx` never reset | per-step counter, unreduced step refused | old counter fails at step 2 | `zwt_ddp_loopback_tests` |
| 4 SIMD CPU kernels cut the graph | autograd path when an input or weight needs a gradient | CPU: 2/30 → 30/30 params trained | `test_grad_coverage cpu` |
| 5 speculative decoding | suffix-causal mask for cached multi-token blocks; exact linear, draft-model and tree steps | 46 checks speculative == greedy; old mask fails 16; context-blind tree accepts 0–4 of 26–30 | `test_spec_greedy` |
| latent | `embed_scale` applied twice; uint16 tokens read as int16; sliding-window decode ignored the window | — | `test_spec_greedy` (window) |
| found while testing | packed-QKV and RoPE caches ignored device moves/reloads; MPS `.to(kCPU, kFloat64)` returned garbage logits; LibTorch 2.14 needs C++20 (autocast probes silently fell back) | real OLMo-2 1B: greedy on MPS printed "iesziesz…" → correct text | `test_spec_greedy mps`, `bench_spec_decode` |

## 2. SL-MTP (shared-logit low-rank MTP heads)

Math: head k is `u_k = a_k h + U_k V_kᵀ h` through the shared final norm and LM head; factored as
`logits_k = (a_k y + T_k c_k) · rsqrt(mean(u_k²)+eps)`, `y = W(g⊙h)` shared, `T_k = W diag(g) U_k`.
Exactness: factored vs direct, all gradients, fp64: ≤ 3e-15 (`test_slmtp`); a planted missing rms(u) gradient is caught (rel err 13).

**Kill test** (frozen OLMo-2-0425-1B, TinyStories, 507k tokens, 1 epoch, best of 3 learning rates per variant;
56k held-out positions). Top-1 at t+2 / t+3 / t+4:

| Heads (3) | Params | Top-1 | Agrees with model's greedy pick | a |
|---|---|---|---|---|
| Full rank (lr 1e-3 of {3e-4, 1e-3, 3e-3}) | 12.6M | 34.2 / 21.2 / 15.6 | 36.7 / 22.1 / 16.2 | — |
| SL-MTP r=512 | 6.3M | 32.6 / 19.8 / 14.5 | 35.7 / 20.8 / 15.0 | 1.00 |
| SL-MTP r=128 | 1.6M | 30.7 / 18.4 / 13.9 | 33.8 / 19.5 / 14.6 | 0.94 |
| SL-MTP r=32 | 0.4M | 27.0 / 15.3 / 11.4 | 30.1 / 16.5 / 12.3 | 0.93 |
| r=128, a pinned 0 | 1.6M | 26.6 / 16.2 / 12.7 | 28.4 / 17.0 / 13.3 | 0 |

**Training cost**, vocabulary side, forward + backward, 3 heads, rank 64 (`bench_slmtp`):

| Hardware | Precision | Shape (d, V, tokens) | Full heads, repo's fused CE | SL-MTP chunked | Change |
|---|---|---|---|---|---|
| M4 Pro | fp32 | 2048, 50304, 1024 | 980 ms | 406 ms | −59% |
| M4 Pro | fp32 | 4096, 100352, 512 | 4,110 ms | 834 ms | −80% |
| A100 | fp32 | 2048, 50304, 2048 | 389 ms | 107 ms | −72% |
| A100 | fp32 | 4096, 100352, 1024 | 1,081 ms | 205 ms | −81% |
| A100 | bf16 | 2048, 50304, 2048 | 96.5 ms | **24.7 ms** (row kernels + mixed-dtype dW) / 73.9 ms (LibTorch) | −74% |
| A100 | bf16 | 4096, 100352, 1024 | 337.6 ms | **35.5 ms** / 110.8 ms | −89% |

Against full heads that keep every logit in memory (fastest possible full-head form, memory grows with
tokens × vocab): bf16 1B 28.4 → 24.7 ms (−13%), 7B 52.6 → 35.5 ms (−32%). The first kernel version (before
mixed-dtype dW accumulation and the one-pass T build) took 37.9 / 77.0 ms. Rank 512 gives no speed gain.
CUDA row kernels match the reference path to 3.2e-7 (fp32), 7e-3 (bf16).

**Decoding** (OLMo-2 1B fp32 on the M4 Pro GPU, rank-128 heads from the kill test, 6 prompts × 128 tokens,
output identical to plain greedy in every run):

| Mode | Tokens per step | Speed vs plain greedy |
|---|---|---|
| linear, 1 draft | 1.29 | 1.08× |
| linear, 2–3 drafts | 1.31–1.32 | 0.98–0.99× |
| tree, fanout 2–3, depth 1–3 | 1.44–1.66 | 0.51–0.61× (two forwards per step today) |

Draft step cost (3 heads): 3.8 → 0.73 ms (1B, 5.2×) and 7.2 → 0.69 ms (7B, 10.5×) on the M4 Pro.

## 3. Factored DoRA (published: arXiv 2603.22276; adopted, not claimed)

`y = (m/‖W+sBA‖)·(xWᵀ + s(xAᵀ)Bᵀ)`, with the row norm from `‖W_i‖² + 2s·B_i·(WAᵀ)_i + s²·B_i(AAᵀ)B_iᵀ`.
No weight-sized tensor is built or kept for the backward, and the frozen base needs no dW′ GEMM.
Equals the materialised form to fp32 precision (both compute the norm in fp32 by design); a planted missing
cross term is caught (`test_dora_factored`).

| Hardware | Shape (tokens, in → out, rank) | Materialised | Factored | Speedup |
|---|---|---|---|---|
| M4 Pro fp32 | 2048, 4096 → 11008, 32 | 143.9 ms | 102.6 ms | 1.40× |
| A100 fp32 | 2048, 4096 → 11008, 32 | 42.5 ms | 22.3 ms | 1.90× |
| A100 bf16 | 2048, 4096 → 11008, 32 | 5.3 ms | 2.9 ms | 1.84× |
| A100 fp32 | 4096, 4096 → 4096, 32 | 23.6 ms | 16.5 ms | 1.43× |
| A100 bf16 | 4096, 4096 → 4096, 32 | 2.9 ms | 2.3 ms | 1.24× |

## 4. Structural tokenizer (repo's "novel tokenizer")

Lossy: 0/103 files round-trip (`test_structural_roundtrip`). It drops indentation, slot whitespace and
boundaries, and all GPT-2 ids ≥ 50,000. Its 9.6% fewer tokens on code cannot be claimed until it is lossless.

## 5. Forge demo on the A100 (7B DoRA SFT, `scripts/colab_forge_demo.sh`)

_Pending: ablation (materialised DoRA + full heads vs factored DoRA vs + SL-MTP), demo training, export._

## 6. Ideas and open problems (not yet measured)

- Fuse the logit GEMM into the SL-MTP row kernel (bf16 already beats materialised full heads; fusion would cut the remaining [tokens × vocab] passes).
- Tree speculation: write the accepted nodes' KV during the verify, removing the second forward per step.
- Train SL-MTP heads longer / with the trunk (acceptance 29% today; ~70% would give ~1.9× decode).
- Make the structural tokenizer lossless (slot boundaries, whitespace, full BPE range), then measure.
- Re-run the 2× H100 DDP baselines (old code never synchronised ranks).
- Metal: the measured missing "free verify window" in llama.cpp (51 → 219 ms for 2 → 7 tokens on 7B q4).
