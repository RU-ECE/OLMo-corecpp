# SL-MTP heads on 16M tokens, Colab A100-SXM4-40GB, 2026-11-20

**Headline (30 prompts, see the last section): 16M-token rank-128 heads 1.178x at draft 2, 48.3% draft-1
acceptance, output identical to greedy; 1.6M-token heads 1.145x / 46.6% on the same GPU. The +1.7 points is
~1.2 standard errors (about 2,600 drafts per cell: SE ~1 point), so not significant. Rank 512 and full-rank
heads add <= 1.5 points of held-out agreement: head capacity is ruled out; whether joint trunk+heads training
helps is the next hypothesis, not a result. The 6-prompt table below (1.240x / 52.2%) is superseded.**

## First run (6 prompts; completes the stopped 2026-11-15 run)

`scripts/colab_slmtp_scale.sh` (this branch's version: refuses to start without Drive or the old heads, saves
as each step ends), TOKENS=16000000. OLMo-2-0425-1B frozen, fp32; TinyStories train split.
- Extract: 31,250 x 512 = 16.0M tokens at 6,133 tok/s; main CE 1.853, top-1 0.557.
- Train: rank 128, self-distilled (--target greedy), batch 2, lr 1e-3, 1 epoch = 14,063 steps on 14,400,512
  tokens, 1,206 s; final train loss 3.481. Held-out agreement with the model's greedy token per head
  (t+2, t+3, t+4): 43.0%, 25.1%, 17.5% (top-1 vs real token 31.9 / 19.1 / 13.5%); learned a = 2.27, 2.00, 1.89.
- Decode: real greedy self-speculation, 6 unseen valid prompts (stories past 20,000), 64-token prompts,
  128 new tokens, every head set on the SAME A100, nothing else on the GPU during decode.

| heads (training text) | draft 1 | draft 2 | draft 3 | identical to greedy |
|---|---|---|---|---|
| heads_sl_r128 (kill test, 0.45M valid) | 1.050x, acc 37.7%, 1.37 tok/fwd | 1.033x, 21.5%, 1.42 | 1.031x, 14.7%, 1.44 | yes |
| heads_sl_r128_big_greedy (1.6M valid, Mac) | 1.097x, 47.9%, 1.47 | 1.194x, 33.2%, 1.66 | 1.188x, 23.2%, 1.69 | yes |
| heads_sl_r128_16000000_greedy (14.4M train) | 1.139x, 52.2%, 1.52 | **1.240x**, 36.7%, 1.73 | 1.227x, 25.4%, 1.75 | yes |

Reading: 9x more text moved draft-1 acceptance 47.9% -> 52.2% and the best speedup 1.194x -> 1.240x on
the same GPU. More data still helps but with sharply diminishing returns, so frozen-trunk heads are near
their ceiling; the plan's ~70% acceptance (~1.9x) needs the trunk to cooperate (joint trunk+heads training,
FastMTP-style). The Mac numbers of 2026-11-17 (1.261x for big_greedy on the M4 Pro) are not comparable:
the same heads give 1.194x on this A100 (the verify/draft cost ratio differs by hardware).
Raw files: Drive llm-cpp-slmtp/ (summary.txt, decode.jsonl, train.json, train.log, extract.log, heads .pt).

## Capacity ablation, same 16M-token features (A100, same evening)

| heads | trainable params | held-out agree_greedy t+2 / t+3 / t+4 | top-1 vs real t+2 | final train loss | train s |
|---|---|---|---|---|---|
| SL-MTP rank 128 (above) | 1,572,867 | 43.0 / 25.1 / 17.5% | 31.9% | 3.481 | 1,206 |
| full rank (d x d proj + RMSNorm, frozen LM head) | 12,589,056 | 44.5 / 26.1 / 18.0% | 32.8% | 3.338 | 2,381 |
| SL-MTP rank 512 | (run queued; a first attempt in parallel was OOM-killed: 62 GB of features x2 > 83 GB RAM) | | | | |

8x the parameters buys +1.5 points of agreement at t+2: head capacity is not what limits acceptance.
With more data also saturating (+4.3 points for 9x text), head capacity is ruled out; joint trunk+heads training (heads can
only predict what the final hidden state already encodes) is the next hypothesis to test, gated on main-head CE 1.853 / top-1 0.557 not degrading.

Rank-512 ablation result: 6,291,459 params, held-out agree_greedy 44.2 / 25.8 / 17.9%, final train loss 3.371,
2,483 s. Ordering r128 43.0 < r512 44.2 < full 44.5: capacity is not the limit.

## 30-prompt rerun (replaces the 6-prompt headline): A100, nothing else on the GPU, 128 new tokens each

| heads | draft 1 | draft 2 | identical |
|---|---|---|---|
| r128, 1.6M valid tokens (Mac-trained) | 1.098x, acc 46.6%, 1.46 tok/fwd | 1.145x, 30.0%, 1.59 | yes |
| r128, 14.4M train tokens | 1.112x, 48.3%, 1.48 | **1.178x**, 32.4%, 1.64 | yes |
| r512, 14.4M train tokens | 1.096x, 49.0%, 1.49 | 1.160x, 33.0%, 1.65 | yes |

KV-cache parity on the real model: max |diff| 1.183e-04, argmax equal.
With 30 prompts the 16M-token heads give 1.178x (best) and 48.3% draft-1 acceptance: the 6-prompt figures
above (1.240x, 52.2%) were optimistic by ~0.06x / ~4 points. 9x more data: +1.7 points (~1.2 SE, not significant), +0.033x.
Speculative-forward cost (implied from tok/fwd / speedup, not timed): tok/fwd 1.64 at 1.178x means each verify step costs ~1.39x a plain forward, so
1.9x would need ~2.6 tokens per forward: acceptance alone cannot get there without cheaper verification.
