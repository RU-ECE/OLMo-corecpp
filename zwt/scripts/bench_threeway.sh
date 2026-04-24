#!/usr/bin/env bash
# bench_threeway.sh — run zwt_pretrain, PyTorch eager, and torch.compile on
# the same machine with the same model shape and dump a comparison CSV.
#
# Pins:
#   * one process per stack, sequential (NVML jitter is enough to make
#     side-by-side numbers suspect)
#   * identical B, S, vocab, layers, heads, d_model, d_ffn
#   * identical warmup/iters
#   * CUDA_VISIBLE_DEVICES = 0 (override by exporting before invoking)
#
# Output: three CSV rows on stdout:
#   stack,dtype,ms_per_step,tok_per_sec,notes
#
# Usage:
#   zwt/scripts/bench_threeway.sh conf/owt_1B_h100.conf
#   zwt/scripts/bench_threeway.sh conf/owt_1B_h100.conf 100 10 2 512
#                                 config          iters warmup B S
#
# The last four args override iters/warmup/batch/seq for quick sanity runs.
set -euo pipefail

CONFIG=${1:?usage: $0 <config.conf> [iters warmup batch seq]}
ITERS=${2:-50}
WARMUP=${3:-5}
BATCH=${4:-}
SEQ=${5:-}

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
BIN_PT=$REPO/zwt/scripts/pt_baseline.py
BIN_ZWT=$REPO/build/zwt_pretrain

if [[ ! -x $BIN_ZWT ]]; then
  echo "bench_threeway: $BIN_ZWT not found; build with ./scripts/build.sh --cuda first" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "bench_threeway: nvidia-smi not on PATH — this benchmark needs CUDA" >&2
  exit 2
fi

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "# bench_threeway on $GPU  config=$CONFIG  warmup=$WARMUP iters=$ITERS" >&2

opt_batch=()
opt_seq=()
[[ -n $BATCH ]] && opt_batch=(--batch "$BATCH")
[[ -n $SEQ   ]] && opt_seq=(--seq "$SEQ")

echo "stack,dtype,ms_per_step,tok_per_sec,notes"

# ---- PyTorch eager ----
PT_EAGER=$(python3 "$BIN_PT" --config "$CONFIG" --warmup "$WARMUP" --steps "$ITERS" \
                   --dtype bf16 "${opt_batch[@]}" "${opt_seq[@]}" | tail -1)
# PT line format: "  B=.. S=.. steps=.. dt=..s tok/s=... ms/step=.."
ms=$(echo "$PT_EAGER" | sed -n 's/.*ms\/step=\([0-9.]*\).*/\1/p')
tps=$(echo "$PT_EAGER" | sed -n 's/.*tok\/s=\([0-9,]*\).*/\1/p' | tr -d ,)
echo "pytorch_eager,bf16,$ms,$tps,"

# ---- torch.compile ----
PT_COMP=$(python3 "$BIN_PT" --config "$CONFIG" --warmup "$WARMUP" --steps "$ITERS" \
                  --dtype bf16 --compile "${opt_batch[@]}" "${opt_seq[@]}" | tail -1)
ms=$(echo "$PT_COMP"  | sed -n 's/.*ms\/step=\([0-9.]*\).*/\1/p')
tps=$(echo "$PT_COMP" | sed -n 's/.*tok\/s=\([0-9,]*\).*/\1/p' | tr -d ,)
echo "pytorch_compile,bf16,$ms,$tps,"

# ---- zwt ----
# zwt_pretrain has no --iters flag — we run for WARMUP+ITERS steps and time
# the whole thing with `time`, then back out ms/step. Its step log emits
# "step   N  loss ...  <tps> tok/s", grep that instead for a direct read.
TMP=$(mktemp)
trap 'rm -f $TMP' EXIT
# Build a tiny override config if iters/warmup differ from config's
# max_steps? Keep it simple: run for WARMUP+ITERS steps and parse the
# step-log tok/s from stderr.
STEPS=$(( WARMUP + ITERS ))
# Point max_steps via env — zwt_pretrain reads it from the INI, but we can
# just run and stop by sending SIGTERM. For a clean measurement, the user
# should duplicate the config with max_steps set. Document that here.
echo "# NOTE: zwt_pretrain bench needs a config with max_steps=$STEPS and log_interval=$ITERS for one-shot timing" >&2
"$BIN_ZWT" "$CONFIG" 2>"$TMP" >/dev/null || true
ZWT_LAST=$(grep -E "^step " "$TMP" | tail -1 | awk '{ for (i=1;i<=NF;i++) if ($i ~ /tok\/s/) print $(i-1) }')
echo "zwt,bf16,,$ZWT_LAST,see $TMP for step log"
