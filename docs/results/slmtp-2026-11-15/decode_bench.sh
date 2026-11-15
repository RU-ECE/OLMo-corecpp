#!/usr/bin/env bash
# Real greedy decode with SL-MTP self-speculation (OLMo-2-0425-1B fp32, M4 Pro GPU), same settings
# as the earlier 1.08x: 6 TinyStories prompts past the training text, 64-token prompts, 128 new tokens.
set -uo pipefail
cd "$(dirname "$0")"
B=../../build/bench_spec_decode
: > decode.jsonl
for heads in heads_sl_r128.pt heads_sl_r128_4ep_real.pt heads_sl_r128_4ep_greedy.pt; do
  for d in 1 2 3; do
    out=$("$B" --config olmo2_1b.json --ckpt olmo2_1b.pt --heads "$heads" --rank 128 --K 3 --draft "$d" \
          --vocab hf/vocab.json --merges hf/merges.txt --text TinyStories-valid.txt \
          --prompts 6 --prompt-len 64 --new 128 --skip 2000 --device mps 2>>decode.log | tail -1)
    echo "{\"heads\": \"$heads\", \"result\": $out}" >> decode.jsonl
  done
done
echo DECODE_DONE >> decode.log
