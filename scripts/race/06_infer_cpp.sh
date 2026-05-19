#!/usr/bin/env bash
# scripts/race/06_infer_cpp.sh
#
# C++ inference race. Uses the chat tool's MTP speculative decoding +
# paged KV cache to generate N tokens from a fixed prompt. Reports
# median tok/s over K trials.
#
# This is where the MTP heads pay off: the linear-chain speculative
# decoder drafts via MTP and verifies, accepting 2-4 tokens per
# verify forward instead of 1.

set -euo pipefail
cd "$(dirname "$0")/../.."

results_dir=scripts/race/results/cpp_infer
mkdir -p "$results_dir"

say() { printf "\033[1;36m[infer-cpp]\033[0m %s\n" "$*"; }

CKPT="${CPP_CKPT:-scripts/race/results/cpp_ckpt/last.pt}"
CONF=scripts/race/configs/race_250m_cpp.conf
VOCAB=data/gpt2/vocab.json
MERGES=data/gpt2/merges.txt

if [[ ! -f "$CKPT" ]]; then
  say "WARN: no C++ checkpoint at $CKPT — pre-trained checkpoint not yet produced"
  say "      run 04_train_cpp.sh first, or set CPP_CKPT=<path> to point at one"
  say "      using last checkpoint from training (if any) — otherwise this will fail"
fi

PROMPT="Once upon a time in a small village by the mountain, there lived"
TRIALS=5
TOKENS=256

say "ckpt:   $CKPT"
say "prompt: $PROMPT"
say "trials: $TRIALS × $TOKENS tokens"

LOG="$results_dir/infer.log"
RESULTS_CSV="$results_dir/results.csv"
echo "trial,tok_per_s,total_tokens,wall_seconds,accept_rate" > "$RESULTS_CSV"

# Five fresh runs (each restarts the chat process so state is clean).
for trial in $(seq 1 "$TRIALS"); do
  say "trial $trial/$TRIALS …"
  trial_log="$results_dir/trial_${trial}.log"
  echo "$PROMPT" | ./build/chat \
      --checkpoint "$CKPT" \
      --config "$CONF" \
      --vocab-file "$VOCAB" \
      --merges-file "$MERGES" \
      --max-tokens "$TOKENS" \
      --temperature 0 \
      --paged-kv \
      --device cuda \
      2>&1 | tee "$trial_log"

  # Extract from the chat tool's trailing stats line:
  #   [256 tokens, 187.4 tok/s, speculative, 64% accepted]
  python3 - <<EOF
import re
with open("$trial_log") as f:
    text = f.read()
m = re.search(r"\[(\d+)\s+tokens,\s+([\d\.]+)\s+tok/s.*?(\d+)%\s+accepted\]", text)
if m:
    tokens, toks, accept = int(m.group(1)), float(m.group(2)), int(m.group(3))
    wall = tokens / toks if toks > 0 else 0.0
    with open("$RESULTS_CSV", "a") as out:
        out.write(f"$trial,{toks},{tokens},{wall:.3f},{accept}\n")
    print(f"  → {toks:.1f} tok/s, accept {accept}%")
else:
    print(f"  (no stats line found in trial $trial — check $trial_log)")
EOF
done

# Summary
python3 - <<EOF
import csv, statistics
rows = list(csv.DictReader(open("$RESULTS_CSV")))
if not rows:
    print("  no successful trials")
else:
    speeds = [float(r["tok_per_s"]) for r in rows]
    accepts = [float(r["accept_rate"]) for r in rows]
    print(f"\n  median tok/s: {statistics.median(speeds):.1f}")
    print(f"  mean   tok/s: {statistics.mean(speeds):.1f}")
    print(f"  min    tok/s: {min(speeds):.1f}")
    print(f"  max    tok/s: {max(speeds):.1f}")
    print(f"  median accept_rate: {statistics.median(accepts):.0f}%")
EOF
