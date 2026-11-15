#!/usr/bin/env bash
# SL-MTP heads, rank 128, 4 epochs: real next tokens vs the model's own greedy picks (self-distillation)
set -uo pipefail
cd "$(dirname "$0")"
T=../../build/slmtp_killtest
COMMON=(--config olmo2_1b.json --ckpt olmo2_1b.pt --features feat --heads 3 --batch 2 --device mps --variant sl --rank 128 --lr 1e-3 --epochs 4)
: > selfdistill.jsonl; : > selfdistill.log
for tgt in real greedy; do
  "$T" train "${COMMON[@]}" --target "$tgt" --save "heads_sl_r128_4ep_$tgt.pt" 2>>selfdistill.log | tee -a selfdistill.jsonl > /dev/null
done
echo SELFDISTILL_DONE >> selfdistill.log
