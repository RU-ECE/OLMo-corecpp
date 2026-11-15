#!/usr/bin/env bash
# SL-MTP data scaling: 4x the features (1.8M tokens), self-distilled heads, decode on unseen prompts.
set -uo pipefail
cd "$(dirname "$0")"
T=../../build/slmtp_killtest; B=../../build/bench_spec_decode
log() { echo "[$(date +%H:%M:%S)] $*" >> scale.log; }
: > scale.log
log "extract 1.8M tokens"
"$T" extract --config olmo2_1b.json --ckpt olmo2_1b.pt --vocab hf/vocab.json --merges hf/merges.txt \
  --text TinyStories-valid.txt --out feat_big --seq 512 --tokens 1800000 --device mps 2>>scale.log || { log FAILED extract; exit 1; }
log "train self-distilled heads, 2 epochs"
"$T" train --config olmo2_1b.json --ckpt olmo2_1b.pt --features feat_big --heads 3 --batch 2 --device mps \
  --variant sl --rank 128 --lr 1e-3 --epochs 2 --target greedy --save heads_sl_r128_big_greedy.pt \
  2>>scale.log > scale_train.json || { log FAILED train; exit 1; }
log "decode on unseen prompts (stories after 20000)"
: > decode_unseen.jsonl
for heads in heads_sl_r128.pt heads_sl_r128_4ep_real.pt heads_sl_r128_4ep_greedy.pt heads_sl_r128_big_greedy.pt; do
  for d in 1 2; do
    out=$("$B" --config olmo2_1b.json --ckpt olmo2_1b.pt --heads "$heads" --rank 128 --K 3 --draft "$d" \
          --vocab hf/vocab.json --merges hf/merges.txt --text TinyStories-valid.txt \
          --prompts 6 --prompt-len 64 --new 128 --skip 20000 --device mps 2>>scale.log | tail -1)
    echo "{\"heads\": \"$heads\", \"result\": $out}" >> decode_unseen.jsonl
  done
done
rm -rf feat_big
log "SCALE_DONE"
