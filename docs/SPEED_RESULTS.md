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
| 6 DoRA finetunes did not start from the base model | `init_weights()` (run by `main.cpp` for every job, finetunes included) drew every 2-D block parameter from N(0, 0.02), DoRA's `lora_B` too, so W + s·BA ≠ W before step 0; adapters now keep their own init (also in µP init) | tiny model: step-0 logits 11% off the base → 1.6e-7; max \|lora_B\| 0.06 → 0 | `test_dora_init` |
| 8 RoPE did nothing on CUDA | all four `kernels/rope.cu` kernels read the cos/sin tables at `[col]`, i.e. position 0 (cos 1, sin 0) for every token, while attention passes one row per position; the backward (ATen) did rotate, so forward and backward disagreed. Kernels now index `(row mod S)·D + col`, with a reference fallback for any other layout | kernel vs reference max \|diff\|: fp32 7.28 → 0, bf16 7.81 → bf16 rounding (equal to rotating everything as position 0). OLMo-2-1124-7B main loss on 4 Forge windows, bf16: CUDA backend 6.559, without it 1.034, Hugging Face 1.039 | `test_cuda_parity` (new RoPE case), `eval_sft_loss --cuda-backend 1`, `scripts/hf_sft_loss.py` |
| 7 checkpoint save held the model in RAM 4–6× | archives serialised into an ostringstream, copied to a string, then to a byte vector (load: the same); `save_async` also cloned on the GPU first. Local checkpoints now stream to/from the file; snapshot copies straight to host | 7B DoRA save on Colab peaked at 69.6 GB host RSS and was OOM-killed. 512 MB model: +3,009 MB (5.9×) → +449 MB (0.9×) | `test_checkpoint_io` (`OLMO_CKPT_BYTES=1` plants the old path) |

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

**Data** (12 gate-passing Forge agent trajectories, ×4 upsampled, + Tulu-3 general conversations at 3:1).
Trajectories are 3k–12k tokens. Sequence length is 4096: OLMo-2-1124-7B's trained context and the cap
Forge's qrt runtime applies (`n_ctx ≤ n_ctx_train`).

| Sequence length | Forge conversations kept (of 48) | Tokens |
|---|---|---|
| 2048 (first attempt) | 0, so the demo would have learned nothing from Forge | 52,765 |
| 4096 | 28 (7 unique) | 145,005 |
| 6144 / 8192 | 44 (11 unique) | 220,917 |
| 12288 | 48 | 260,761 |

Packing fix: llm-cpp trains on fixed windows `tokens[k·S : k·S+S]`. Concatenated conversations cut one at
**34 of 34** window boundaries, so the second half of most trajectories trained without its system
prompt and task, and 1,203 assistant tokens after the last window were never trained. Whole-conversation
bin packing (`prep_forge_sft.py --layout windows`, Forge repo): 0 cuts, 0 lost, 1.7% padding, 36 windows,
85,133 trained tokens (`check_sft_windows.py`, which fails on the old layout).

**Ablation** (OLMo-2-1124-7B, DoRA r=32 on all 7 projections, pure bf16, full activation checkpointing,
4 × 4096 tokens per step, same data order; 8 steps each, steady-state = median of steps 2–7):

| Configuration | Step | Tokens/s | Speedup | Logged loss after 7 steps |
|---|---|---|---|---|
| Materialised DoRA + 2 full-rank MTP heads (the repo's recipe) | 7,866 ms | 2,083 | 1.00× | 2.376 |
| Factored DoRA + 2 full-rank heads | 6,402 ms | 2,559 | **1.23×** | 2.375 |
| Factored DoRA + 3 SL-MTP heads (r=128, chunked loss, row kernels) | 6,142 ms | 2,668 | **1.28×** | 2.541 (3 heads, not comparable) |

Re-run with all fixes (2026-11-15 00:54); the pre-fix run's step times agree within 0.3%. The SL-MTP run trains
one more head and is still the fastest. Logged loss = main CE + 0.1 × mean head CE, so it is not comparable
across head types. Estimated utilisation ≈ 38% of A100 bf16 peak (FLOP count: ≈ 45 GFLOP per token with
full recompute and a frozen base). nvidia-smi peaks (31–40 GB) track the caching allocator, not need, and
changed between identical runs, so no memory claim is made.

**Correctness check:** after the RoPE and DoRA-init fixes, the trainer's own main loss (no MTP heads, lr 0,
one epoch of the 36 windows) is **0.9648** against Hugging Face's **0.9657** (before: 6.5314).

**Demo model v0** (base OLMo-2-1124-7B, ChatML, 27 steps = 3 epochs, LM head frozen, Q4_K_M 4.47 GB exported).
Hugging Face on the same windows: main loss 0.966 → **0.839** (mean per window); `<|im_end|>` 24.6 → 11.5 nats.
Greedy reply to a Forge task: the base model repeats the prompt; the demo emits a correct `knowsy_scaffold`
tool call (name, category, widget, prefix) but never ends its turn, because a base model has never produced
`<|im_end|>` and low-rank adapters barely move that output row.

| Start (same 7 Forge conversations, 27 steps; Hugging Face scores) | Main loss, mean per window | Turn-end token (nats) | Greedy reply stops |
|---|---|---|---|
| Base, untrained (ChatML) | 0.966 | 24.6 | no (repeats the prompt) |
| Base, DoRA, LM head frozen | 0.839 | 11.5 | no: correct tool call, then loops |
| Base, DoRA + LM head trained (`train_lm_head`, +2% step time) | 0.616 | 5.96 | no: complete tool call, then loops |
| OLMo-2-1124-7B-**Instruct**, untrained (its own Tulu format) | 0.682 | 1.93 | n/a (writes code instead of a tool call) |
| **Instruct, DoRA, 27 steps** | **0.486** | **0.28** | **yes**, after 102 tokens: a `knowsy_scaffold` call, then `<\|endoftext\|>` |

The trained Instruct reply has the right tool and widget id/category but drops `class_name` and `prefix` and
adds junk keys: format and stopping are solved, argument correctness needs more than 7 conversations.

Untrained Instruct is already near the trained base model and knows its turn end, so it is the starting point
for the real demo (`BASE_MODEL=allenai/OLMo-2-1124-7B-Instruct`). Export (merge 14.5 min, HF write ~13 min,
GGUF + Q4_K_M ~10 min) now takes longer than training; writing the merge in bf16 would halve the first two.

**Limit of this data:** 6 tasks, 12 passing runs, 7 that fit 4096 tokens. A usable agent needs more: 46 new
tasks (`bench/knowsybench/make_tasks.py`, 38 for training, 8 held out), trajectories generated with
Qwen2.5-Coder-32B-Instruct and kept only when all three Forge gates pass, then scoring on the held-out tasks
with the same gates.

## 6. Ideas and open problems (not yet measured)

- Fuse the logit GEMM into the SL-MTP row kernel (bf16 already beats materialised full heads; fusion would cut the remaining [tokens × vocab] passes).
- Tree speculation: write the accepted nodes' KV during the verify, removing the second forward per step.
- Train SL-MTP heads longer / with the trunk (acceptance 29% today; ~70% would give ~1.9× decode).
- Make the structural tokenizer lossless (slot boundaries, whitespace, full BPE range), then measure.
- Re-run the 2× H100 DDP baselines (old code never synchronised ranks).
- Metal: the measured missing "free verify window" in llama.cpp (51 → 219 ms for 2 → 7 tokens on 7B q4).
