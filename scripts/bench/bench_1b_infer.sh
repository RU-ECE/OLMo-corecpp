#!/usr/bin/env bash
# scripts/bench/bench_1b_infer.sh
#
# Inference benchmark for the trained 1B: OUR C++ engine vs ollama vs llama.cpp,
# at matched 1B scale, same prompt shape, on whatever GPU this box has.
#
#   - C++  : bench_chat on YOUR model.pt (bf16, + INT4 if you quantize it),
#            batch 1 (single-stream) and 8/32 (throughput — our edge).
#   - ollama  : a standard ~1B (llama3.2:1b, Q4) single-stream eval rate.
#   - llama.cpp: same standard 1B GGUF via llama-bench (or llama-cli).
#
# Why a standard 1B for ollama/llama.cpp: they need GGUF, and converting our
# custom-arch checkpoint to GGUF is fragile (MTP heads, reordered norm). Matching
# at 1B *scale* measures engine speed fairly — same FLOPs/token. (To compare the
# EXACT same weights, see the optional GGUF-export section in PROFESSOR_INFERENCE.md.)
#
# Usage:
#   CKPT=runs/kuiper_1B/model.pt CONFIG=configs/kuiper_1B.json \
#     bash scripts/bench/bench_1b_infer.sh
#   DEVICE=cuda PROMPT_LEN=128 DECODE_LEN=256 bash scripts/bench/bench_1b_infer.sh

set -uo pipefail
cd "$(dirname "$0")/../.."
say(){ echo -e "\n=== $* ==="; }

BUILD_DIR="${BUILD_DIR:-build}"
CKPT="${CKPT:-runs/kuiper_1B/model.pt}"
CONFIG="${CONFIG:-configs/kuiper_1B.json}"
VOCAB="${VOCAB:-data/gpt2/vocab.json}"
MERGES="${MERGES:-data/gpt2/merges.txt}"
DEVICE="${DEVICE:-cuda}"                 # cuda | mps | cpu
PROMPT_LEN="${PROMPT_LEN:-128}"
DECODE_LEN="${DECODE_LEN:-256}"
BATCHES="${BATCHES:-1 8 32}"
OLLAMA_TAG="${OLLAMA_TAG:-llama3.2:1b}"  # standard ~1B baseline (Q4)
LLAMACPP_GGUF="${LLAMACPP_GGUF:-}"       # path to a 1B .gguf for llama.cpp (optional)

[[ -x "$BUILD_DIR/bench_chat" ]] || { echo "ERROR: $BUILD_DIR/bench_chat missing — ./scripts/build.sh --cuda"; exit 1; }
[[ -f "$CKPT" ]]   || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG"; exit 1; }
[[ -f "$VOCAB" ]]  || { mkdir -p "$(dirname "$VOCAB")"; curl -fsSL https://huggingface.co/gpt2/resolve/main/vocab.json -o "$VOCAB"; }
[[ -f "$MERGES" ]] || curl -fsSL https://huggingface.co/gpt2/resolve/main/merges.txt -o "$MERGES"

# Optional: build an INT4 copy for the fair-vs-Q4 single-stream row.
CKPT_INT4=""
if [[ "${MAKE_INT4:-1}" == "1" && -x "$BUILD_DIR/quantize_int4" ]]; then
  CKPT_INT4="${CKPT%.pt}.int4.pt"
  if [[ ! -f "$CKPT_INT4" ]]; then
    say "quantizing INT4 -> $CKPT_INT4"
    "$BUILD_DIR/quantize_int4" --in "$CKPT" --out "$CKPT_INT4" --config "$CONFIG" || CKPT_INT4=""
  fi
fi

bench_cpp(){ # $1=label $2=ckpt $3=batch
  "$BUILD_DIR/bench_chat" --checkpoint "$2" --config "$CONFIG" \
    --vocab-file "$VOCAB" --merges-file "$MERGES" --device "$DEVICE" \
    --prompt-len "$PROMPT_LEN" --decode-len "$DECODE_LEN" --batch "$3" --warmup 1 --iters 3 2>/dev/null \
    | grep -iE "Throughput|TPOT|TTFT" | sed "s/^/   [$1 b=$3] /"
}

say "1. OUR C++ engine (bench_chat) — your trained 1B, device=$DEVICE"
for b in $BATCHES; do
  echo "-- bf16, batch $b --"; bench_cpp bf16 "$CKPT" "$b"
  [[ -n "$CKPT_INT4" ]] && { echo "-- int4, batch $b --"; bench_cpp int4 "$CKPT_INT4" "$b"; }
done

say "2. ollama ($OLLAMA_TAG) — single-stream baseline at 1B scale"
OLLAMA_TPS="n/a"
if command -v ollama >/dev/null; then
  ollama pull "$OLLAMA_TAG" >/dev/null 2>&1 || echo "   (pull failed; set OLLAMA_TAG to an installed 1B)"
  OLLAMA_TPS=$(printf 'Write a short paragraph about the ocean.' \
    | ollama run "$OLLAMA_TAG" --verbose 2>&1 | grep -iE "eval rate" | grep -oE "[0-9.]+ tokens/s" | head -1)
  echo "   ollama eval rate: ${OLLAMA_TPS:-<not parsed>}"
else
  echo "   ollama not installed (https://ollama.com) — skipping"
fi

say "3. llama.cpp — single-stream baseline at 1B scale"
LLAMA_TPS="n/a"
LLAMA_BIN="$(command -v llama-bench || command -v llama-cli || true)"
if [[ -n "$LLAMA_BIN" && -n "$LLAMACPP_GGUF" && -f "$LLAMACPP_GGUF" ]]; then
  if [[ "$LLAMA_BIN" == *llama-bench ]]; then
    LLAMA_TPS=$("$LLAMA_BIN" -m "$LLAMACPP_GGUF" -p "$PROMPT_LEN" -n "$DECODE_LEN" 2>/dev/null | grep -iE "tg|tokens" | tail -1)
  else
    LLAMA_TPS=$("$LLAMA_BIN" -m "$LLAMACPP_GGUF" -n "$DECODE_LEN" -p "ocean" 2>&1 | grep -iE "tokens per second|eval time" | tail -1)
  fi
  echo "   llama.cpp: ${LLAMA_TPS:-<not parsed>}"
else
  echo "   llama.cpp not found or LLAMACPP_GGUF unset — skipping."
  echo "   (build llama.cpp, pull a 1B GGUF e.g. bartowski/Llama-3.2-1B-Instruct-GGUF,"
  echo "    then: LLAMACPP_GGUF=/path/Llama-3.2-1B-Q4_K_M.gguf bash $0 )"
fi

say "SUMMARY"
echo "  Our C++ batch-1 row = single-stream latency (compare vs ollama/llama.cpp above)."
echo "  Our C++ batch-8/32 rows = THROUGHPUT — the C++ engine's edge ollama can't match single-stream."
echo "  Fair single-stream comparison: our INT4 vs ollama/llama.cpp Q4 (both 4-bit)."
echo "  ollama single-stream: ${OLLAMA_TPS:-n/a}   llama.cpp: ${LLAMA_TPS:-n/a}"
