# OLMo-corecpp — Measured Results (for the paper)

All numbers pulled from the actual run logs on the project servers (2× H100 box
`149.165.174.207` and kuiper 1× H100 box `149.165.168.67`). **Honest reporting:**
where the C++ engine does not beat PyTorch, it says so.

## 1. C++ engine vs Python OLMo-core — head-to-head (the headline)
Identical model (768d / 12L / 12H), batch 8, seq 512, 100 steps, single H100.

| Engine | Avg step | Reported throughput | Steady-state step | Wall (100 steps) |
|---|---|---|---|---|
| **C++ (fused, GPU-resident data, ForeachAdamW)** | 85.9 ms | 47,661 tok/s | **~67–81 ms** | 8.59 s |
| **Python (PyTorch, AdamW, bf16 autocast)** | 76.6 ms | 53,466 tok/s | ~75–79 ms | 7.66 s |

**Honest read:** on this 100-step micro-benchmark the C++ engine's *average* throughput
is ~11% **lower** than PyTorch, because step 0 takes **1357 ms** (CUDA-graph capture +
kernel JIT) and drags the 100-step average. Looking at **steady-state** (steps 20–90),
the C++ step time (~67–81 ms) is **comparable to / slightly faster than** PyTorch (~77 ms).
The warmup amortizes over a real (thousands-of-steps) run, so **at scale the engines are
on par**, not 5–20×. *The "5–20× speedup" framing is not supported by this H100 training
benchmark and should be scoped to where the wins actually are (below) or revised.*

Where the C++ engine genuinely wins: **zero per-step H2D** (GPU-resident data loader),
**fused kernels** (RMSNorm / SiLU·mul / RoPE / fused-QKV / fused-gate-up), **GPU-resident
grad clipping** (foreach_norm, no D2H sync), **deterministic low-overhead steps**, and
**inference CUDA graphs** (§5).

## 2. Training throughput by model size (single H100, C++ engine)
| Run | Params | step_ms | Throughput | Steps | Status |
|---|---|---|---|---|---|
| shader_125M | 152M (768d/12L) | 122 ms | **266,979 tok/s** | 4000 | completed |
| shader_v2 | 152M | 122 ms | 267,451 tok/s | 8000/10000 | **partial** |
| shader_1B | ~1B | — | — | <10000 | **partial (interrupted)** |
| shader_3B | ~3B | — | — | <20000 | **partial (interrupted, needed FSDP)** |

(125M sustains ~267k tok/s on one H100 with the large GPU-resident batch — that is the
real sustained-throughput figure to cite.)

## 3. Multi-GPU (2× H100, NCCL DDP)
| Model | step_ms | Throughput | Steps | Note |
|---|---|---|---|---|
| 1,577M (d=2048, 24L) | ~1213 ms | ~53,253 tok/s | 9850/10000 | DDP across 2 ranks; one rank failed near the end |

## 4. OLMo-2-7B DoRA + MTP finetune (kuiper, 1× H100)
The instruction finetune (SFT on Tülu-3, ChatML) done entirely on the C++ engine.

| Metric | Value |
|---|---|
| Trainable params | **114.876M / 7,413.5M (1.55%)** — base frozen (DoRA + 2 MTP heads) |
| Steps / epochs | 2500 / 5 |
| Total tokens | 81,920,000 |
| Wall time | **27,319 s (~7.6 h)** |
| Throughput | **2,998 tok/s** (step_ms ~10,800; full activation checkpointing, batch 1, seq 2048) |
| Loss | 7.4 → **2.6** (epoch avg 4.00 → 2.73) |

## 5. Inference (OLMo-2-7B, 1× H100, C++ engine)
| Path | Throughput | Note |
|---|---|---|
| Base, eager | ~7.2 tok/s | baseline |
| Base, **CUDA graphs + paged-KV** | **47.9 tok/s** | **6.6× over eager** |
| Merged finetuned model (fp32) | 7.1 tok/s | fp32 29GB, memory-bound; bf16/INT4 in progress |
| MTP self-speculative decoding | — | **28–31% draft acceptance** (one head drafts, model verifies) |

## 6. Optimizations enabled (per the configs/logs)
Fused kernels; GPU-resident data loading (zero per-step H2D); ForeachAdamW (batched
`_foreach_*`); GPU-resident grad clipping; arena allocator; CUDA graphs (inference);
NCCL DDP (multi-GPU); activation checkpointing (7B finetune); DoRA + MTP retrofit.

## 7. Partial / unfinished runs (flagged honestly)
- `shader_1B` (12G) — interrupted before its 10k-step target, no summary.
- `shader_3B` (28G) — interrupted before its 20k-step target (this is the 3B that needs
  FSDP; FSDP is wired in code but not yet validated end-to-end on 2 GPUs).
- `shader_v2` — stopped at 8000/10000.
- `dora32b_out` — empty (32B DoRA attempt produced no output).

---
### Bottom line for the paper
- **Sustained training throughput** (e.g. 267k tok/s @ 152M on one H100) and the
  **6.6× inference speedup via CUDA graphs** are the solid, defensible wins.
- The **head-to-head vs PyTorch on H100 training is roughly on par** (warmup-dragged
  average looks ~11% slower; steady-state comparable) — **not 5–20×.** Cite the real
  steady-state and the architectural wins; don't claim a multiplier the data doesn't show.
- A complete **OLMo-2-7B instruction finetune** (DoRA+MTP, 1.55% trainable, ~7.6 h,
  loss 7.4→2.6) ran end-to-end on the C++ engine — a strong systems result on its own.
