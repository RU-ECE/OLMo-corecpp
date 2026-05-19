#!/usr/bin/env bash
# scripts/race/04_train_cpp.sh
#
# Train the C++ 250M (backbone + MTP) for the step count specified in
# the conf. Captures step times, loss, and tok/s into a structured log.

set -euo pipefail
cd "$(dirname "$0")/../.."

results_dir=scripts/race/results/cpp_train
mkdir -p "$results_dir"

say() { printf "\033[1;36m[cpp]\033[0m %s\n" "$*"; }

CONF=scripts/race/configs/race_250m_cpp.conf
LOG="$results_dir/train.log"
METRICS="$results_dir/metrics.csv"

say "config: $CONF"
say "log:    $LOG"

# Wrap the train binary so we capture wall time + GPU utilization.
start=$(date +%s)
{
  echo "[run] $(date -u +%FT%TZ)"
  echo "[host] $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "[conf] $CONF"
  echo
  ./build/olmo_train "$CONF"
} 2>&1 | tee "$LOG"
end=$(date +%s)

say "wall clock: $((end - start)) s"

# Extract step-time + loss + tok/s from the log into a CSV the analyzer reads.
# olmo_train prints lines like:
#   step  100 | loss 5.234 | step_ms 152.3 | tok/s 13456
python3 - <<EOF
import re, csv
rows = []
with open("$LOG") as f:
    for line in f:
        m = re.search(r"step\s+(\d+).*?loss\s+([\d\.eE+-]+).*?step_ms\s+([\d\.]+).*?tok/s\s+([\d\.]+)", line)
        if m:
            rows.append({"step": int(m.group(1)),
                         "loss": float(m.group(2)),
                         "step_ms": float(m.group(3)),
                         "tok_per_s": float(m.group(4))})

with open("$METRICS", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["step","loss","step_ms","tok_per_s"])
    w.writeheader()
    w.writerows(rows)

if rows:
    last = rows[-1]
    print(f"  {len(rows)} step records; final: step={last['step']} loss={last['loss']:.4f} "
          f"step_ms={last['step_ms']:.1f} tok/s={last['tok_per_s']:.0f}")
EOF
