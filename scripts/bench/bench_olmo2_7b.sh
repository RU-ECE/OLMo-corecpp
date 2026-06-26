#!/usr/bin/env bash
# scripts/bench/bench_olmo2_7b.sh
#
# Inference benchmark for OLMo-2-7B: OUR C++ engine vs ollama vs llama.cpp, the
# SAME model (allenai/OLMo-2-1124-7B), same prompt shape, on this H100.
#
#   - C++  : bench_chat on our converted olmo2_7b.pt at fp32 / bf16 / INT4 /
#            +CUDA-graphs, batch 1 (single-stream) and 8/32 (throughput).
#   - ollama  : olmo2:7b single-stream (--verbose eval rate).
#   - llama.cpp: same model GGUF (set LLAMACPP_GGUF) single-stream.
#
# Flags --bf16 / --int4 / --cuda-graph come from the speed work; this harness
# skips any row whose tool/flag isn't ready yet, so it's safe to run early.
#
# Usage (on kuiper):
#   bash scripts/bench/bench_olmo2_7b.sh
#   OLLAMA_TAG=olmo2:7b LLAMACPP_GGUF=/path/OLMo-2-7B-Q4_K_M.gguf bash scripts/bench/bench_olmo2_7b.sh

set -uo pipefail
cd "$(dirname "$0")/../.."
say(){ echo -e "\n=== $* ==="; }

BUILD="${BUILD_DIR:-build}"
V="${VOL:-/media/volume/Prep_and_Voice_Training}"
CKPT="${CKPT:-$V/olmo2-7b/olmo2_7b.pt}"
CKPT_INT4="${CKPT_INT4:-$V/olmo2-7b/olmo2_7b.int4.pt}"
CFG="${CFG:-configs/olmo2_1124_7B.json}"
VOCAB="$V/olmo2-tok/vocab.json"; MERGES="$V/olmo2-tok/merges.txt"
DEVICE="${DEVICE:-cuda}"; PLEN="${PLEN:-64}"; DLEN="${DLEN:-128}"
OLLAMA_TAG="${OLLAMA_TAG:-olmo2:7b}"
LLAMACPP_GGUF="${LLAMACPP_GGUF:-}"

[[ -x "$BUILD/bench_chat" ]] || { echo "build bench_chat first"; exit 1; }
B="$BUILD/bench_chat"
common="--config $CFG --vocab-file $VOCAB --merges-file $MERGES --device $DEVICE --prompt-len $PLEN --decode-len $DLEN --warmup 0 --iters 2"
run(){ "$B" --checkpoint "$1" $common "${@:2}" 2>/dev/null | grep -iE "Throughput" | sed "s/^/   /"; }

say "1. OUR C++ ENGINE (OLMo-2-7B)"
echo "-- fp32, batch 1 --";          run "$CKPT" --batch 1
echo "-- bf16, batch 1 --";          run "$CKPT" --batch 1 --bf16
[[ -f "$CKPT_INT4" ]] && { echo "-- INT4, batch 1 --"; run "$CKPT_INT4" --batch 1 --int4; }
echo "-- fp32 + CUDA graph, batch 1 --"; run "$CKPT" --batch 1 --cuda-graph
echo "-- throughput: bf16 batch 8 / 32 --"; run "$CKPT" --batch 8 --bf16; run "$CKPT" --batch 32 --bf16

say "2. ollama ($OLLAMA_TAG) single-stream"
OLLAMA_TPS="n/a"
if command -v ollama >/dev/null; then
  ollama pull "$OLLAMA_TAG" >/dev/null 2>&1 || echo "   (pull failed; set OLLAMA_TAG)"
  OLLAMA_TPS=$(printf 'Write one paragraph about the ocean.' | ollama run "$OLLAMA_TAG" --verbose 2>&1 \
    | grep -iE "eval rate" | grep -oE "[0-9.]+ tokens/s" | head -1)
  echo "   ollama: ${OLLAMA_TPS:-<not parsed>}"
else echo "   ollama not installed — skipping"; fi

say "3. llama.cpp single-stream"
LB="$(command -v llama-bench || command -v llama-cli || true)"
if [[ -n "$LB" && -n "$LLAMACPP_GGUF" && -f "$LLAMACPP_GGUF" ]]; then
  "$LB" -m "$LLAMACPP_GGUF" -p "$PLEN" -n "$DLEN" 2>/dev/null | grep -iE "tg|tokens|t/s" | tail -2
else echo "   set LLAMACPP_GGUF=/path/to/OLMo-2-7B.gguf (+ build llama.cpp) — skipping"; fi

say "READ ME"
echo "  batch-1 = single-stream latency: compare our INT4 vs ollama/llama.cpp Q4 (both 4-bit)."
echo "  batch-8/32 = throughput: our batched edge; report aggregate tok/s."
echo "  +CUDA graph row is the headline single-stream number (kills eager launch overhead)."
echo "  ollama single-stream: ${OLLAMA_TPS:-n/a}"
