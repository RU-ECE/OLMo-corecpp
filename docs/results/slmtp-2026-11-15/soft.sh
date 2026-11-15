#!/usr/bin/env bash
# Soft self-distillation (the model's whole distribution) vs the greedy-pick target, same data and epochs.
set -uo pipefail
cd "$(dirname "$0")"
T=../../build/slmtp_killtest; B=../../build/bench_spec_decode
: > soft.log
"$T" train --config olmo2_1b.json --ckpt olmo2_1b.pt --features feat --heads 3 --batch 2 --device mps \
  --variant sl --rank 128 --lr 1e-3 --epochs 4 --target soft --save heads_sl_r128_4ep_soft.pt \
  2>>soft.log > soft_train.json || { echo FAILED train >> soft.log; exit 1; }
: > decode_soft.jsonl
for d in 1 2; do
  out=$("$B" --config olmo2_1b.json --ckpt olmo2_1b.pt --heads heads_sl_r128_4ep_soft.pt --rank 128 --K 3 --draft "$d" \
        --vocab hf/vocab.json --merges hf/merges.txt --text TinyStories-valid.txt \
        --prompts 6 --prompt-len 64 --new 128 --skip 20000 --device mps 2>>soft.log | tail -1)
  echo "{\"heads\": \"heads_sl_r128_4ep_soft.pt\", \"result\": $out}" >> decode_soft.jsonl
done
echo SOFT_DONE >> soft.log
