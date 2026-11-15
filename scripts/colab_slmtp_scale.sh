#!/usr/bin/env bash
# SL-MTP draft heads at scale on one Colab A100: does more text keep raising acceptance?
#
#   /content/llm-cpp            llm-cpp source
#   /content/killtest           olmo2_1b.json + earlier heads (heads_sl_r128.pt, heads_sl_r128_big_greedy.pt)
#   bash /content/llm-cpp/scripts/colab_slmtp_scale.sh
#
# Frozen OLMo-2-0425-1B, rank-128 SL-MTP heads trained on the model's own greedy picks
# (slmtp_killtest --target greedy) over TOKENS tokens of the TinyStories train split (the Mac runs
# used 0.45M and 1.6M tokens of the valid split), then real greedy decoding with self-speculation on
# the same unseen valid prompts (stories past 20,000) for the old and new heads, on this GPU.
set -uo pipefail
L=${LLMCPP:-/content/llm-cpp}; K=${KILLTEST:-/content/killtest}
W=${WORK:-/mnt/local-scratch/slmtp}; R=$W/results
TOKENS=${TOKENS:-16000000}
mkdir -p "$W" "$R"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$R/run.log"; }
fail() { log "FAILED: $*"; exit 1; }
B=$L/build-cuda
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
log "GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); TOKENS $TOKENS"

if [ ! -x "$B/slmtp_killtest" ] || [ ! -x "$B/bench_spec_decode" ]; then
  log "build"
  pip -q install ninja > /dev/null 2>&1 || true
  PREFIX=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
  cmake -G Ninja -S "$L" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DOLMO_RACE_BUILD=OFF > "$R/build.log" 2>&1 || fail "configure"
  ninja -C "$B" -j"$(nproc)" slmtp_killtest bench_spec_decode convert_hf >> "$R/build.log" 2>&1 \
    || { grep -n -A3 error "$R/build.log" | head -20; fail "build"; }
fi

if [ ! -s "$W/olmo2_1b.pt" ]; then
  log "model: allenai/OLMo-2-0425-1B -> llm-cpp fp32"
  python3 - <<PY || fail "model download"
from huggingface_hub import snapshot_download
snapshot_download("allenai/OLMo-2-0425-1B", local_dir="$W/hf",
                  allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json"])
PY
  "$B/convert_hf" --hf-dir "$W/hf" --config "$K/olmo2_1b.json" --output "$W/olmo2_1b.pt" --fp32 > "$R/convert.log" 2>&1 \
    || { tail -5 "$R/convert.log"; fail "convert"; }
fi

if [ ! -s "$W/TinyStories-train.txt" ]; then
  log "data: TinyStories train + valid"
  python3 - <<PY || fail "data download"
from huggingface_hub import hf_hub_download
for f in ["TinyStories-train.txt", "TinyStories-valid.txt"]:
    hf_hub_download("roneneldan/TinyStories", f, repo_type="dataset", local_dir="$W")
PY
fi

FEAT=$W/feat_${TOKENS}
if [ ! -s "$FEAT/meta.json" ]; then
  log "extract $TOKENS tokens (train split)"
  "$B/slmtp_killtest" extract --config "$K/olmo2_1b.json" --ckpt "$W/olmo2_1b.pt" --vocab "$W/hf/vocab.json" \
    --merges "$W/hf/merges.txt" --text "$W/TinyStories-train.txt" --out "$FEAT" --seq 512 --tokens "$TOKENS" \
    --device cuda > "$R/extract.json" 2> "$R/extract.log" || { tail -5 "$R/extract.log"; fail "extract"; }
fi

NEW=$W/heads_sl_r128_${TOKENS}_greedy.pt
if [ ! -s "$NEW" ]; then
  log "train self-distilled heads (rank 128, 1 epoch, batch 2, lr 1e-3: the Mac recipe)"
  "$B/slmtp_killtest" train --config "$K/olmo2_1b.json" --ckpt "$W/olmo2_1b.pt" --features "$FEAT" --heads 3 \
    --batch 2 --device cuda --variant sl --rank 128 --lr 1e-3 --epochs 1 --target greedy --save "$NEW" \
    > "$R/train.json" 2> "$R/train.log" || { tail -5 "$R/train.log"; fail "train"; }
fi

log "decode: old and new heads on the unseen valid prompts"
: > "$R/decode.jsonl"
for heads in "$K/heads_sl_r128.pt" "$K/heads_sl_r128_big_greedy.pt" "$NEW"; do
  for d in 1 2 3; do
    out=$("$B/bench_spec_decode" --config "$K/olmo2_1b.json" --ckpt "$W/olmo2_1b.pt" --heads "$heads" --rank 128 \
          --K 3 --draft "$d" --vocab "$W/hf/vocab.json" --merges "$W/hf/merges.txt" --text "$W/TinyStories-valid.txt" \
          --prompts 6 --prompt-len 64 --new 128 --skip 20000 --device cuda 2>> "$R/decode.log" | tail -1)
    echo "{\"heads\": \"$(basename "$heads")\", \"result\": $out}" >> "$R/decode.jsonl"
  done
done
python3 - <<PY | tee "$R/summary.txt"
import json
t = json.load(open("$R/train.json"))
print("new heads:", t.get("train_tokens"), "train tokens; agree_greedy",
      [round(h["agree_greedy"] * 100, 1) for h in t["after"]], "a", [round(x, 2) for x in t["a"]])
for l in open("$R/decode.jsonl"):
    j = json.loads(l); r = j["result"]
    print(f"{j['heads']:40} draft {r['draft']}  speedup {r['speedup']:.3f}  accept {r['acceptance']:.3f}  "
          f"tok/fwd {r['tokens_per_forward']:.2f}  identical {r['outputs_identical']}")
PY
[ -d /content/drive/MyDrive ] && mkdir -p /content/drive/MyDrive/llm-cpp-slmtp && cp "$R"/* "$NEW" /content/drive/MyDrive/llm-cpp-slmtp/ 2>/dev/null
log "SLMTP_SCALE_DONE"
