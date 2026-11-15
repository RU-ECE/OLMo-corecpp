# Colab A100-SXM4-80GB, 16M-token SL-MTP run (stopped early, 2026-11-15)

`scripts/colab_slmtp_scale.sh` with TOKENS=16000000, stopped by request at step 6,000 of 14,063
before evaluation and decoding, so there is no acceptance or speedup number from it.

- Build 8 min; OLMo-2-0425-1B converted to llm-cpp fp32; TinyStories train split.
- Extract: 31,250 sequences × 512 (16.0M tokens), 6,140 tok/s (batch 1, fp32), main CE 1.8532, top-1 0.5569.
- Train (rank 128, self-distilled, batch 2, lr 1e-3, 1 epoch = 14,063 steps): 0.08 s/step on the A100
  (vs 0.47 s on the M4 Pro). Loss at steps 1k/3k/4k/5k/6k: 3.957 / 3.604 / 3.247 / 3.733 / 3.625.
- To finish it: rerun the script (about 75 min on an A100: 50 extract, 19 train, 10 decode).
