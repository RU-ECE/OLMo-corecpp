#!/usr/bin/env bash
# scripts/race/05_train_python.sh
#
# Train the Python 250M backbone (no MTP) using OLMo-core's training
# entry point. Captures step times, loss, and tok/s into a structured log.
#
# Per CLAUDE.md feedback: invoke upstream as a black box. We try
# OLMo-core's standard entry point; if it doesn't match what's checked
# in, the user just edits the OLMO_TRAIN_CMD env var below.
#
# Default expectation:
#   OLMo-corecpp/  — the reference Python OLMo-core repo
#   python entry point at OLMo-corecpp/src/scripts/train.py
#
# Override with:
#   OLMO_TRAIN_CMD="python -m olmo_core.train" bash 05_train_python.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

results_dir=scripts/race/results/python_train
mkdir -p "$results_dir"

say() { printf "\033[1;36m[py]\033[0m %s\n" "$*"; }
fail(){ printf "\033[1;31m[py]\033[0m %s\n" "$*"; exit 1; }

CONF=scripts/race/configs/race_250m_python.yaml
LOG="$results_dir/train.log"
METRICS="$results_dir/metrics.csv"

# Locate the Python OLMo-core training entry.
if [[ -n "${OLMO_TRAIN_CMD:-}" ]]; then
  CMD="$OLMO_TRAIN_CMD"
elif [[ -f OLMo-corecpp/src/scripts/train.py ]]; then
  CMD="python3 OLMo-corecpp/src/scripts/train.py"
elif [[ -f OLMo-corecpp/scripts/train.py ]]; then
  CMD="python3 OLMo-corecpp/scripts/train.py"
elif python3 -c "import olmo_core" 2>/dev/null; then
  CMD="python3 -m olmo_core.train"
else
  fail "OLMo-core training entry not found.
       Set OLMO_TRAIN_CMD='<your invocation>' and re-run.
       e.g.  export OLMO_TRAIN_CMD='python -m olmo_core.train'"
fi

say "command: $CMD"
say "config:  $CONF"
say "log:     $LOG"

start=$(date +%s)
{
  echo "[run] $(date -u +%FT%TZ)"
  echo "[cmd] $CMD"
  echo "[conf] $CONF"
  echo
  $CMD "$CONF"
} 2>&1 | tee "$LOG"
end=$(date +%s)

say "wall clock: $((end - start)) s"

# OLMo-core's logging format varies. We try to extract step / loss / tok-per-s
# by a few common patterns; users can override the regex by setting OLMO_LOG_REGEX.
python3 - <<EOF
import re, csv, os

regex = os.environ.get("OLMO_LOG_REGEX",
    r"(?:step|Step)\s*[: =]\s*(\d+).*?loss[:\s=]+([\d\.eE+-]+)"
    r"(?:.*?(?:throughput|tokens_per_sec|tok/?s)[:\s=]+([\d\.]+))?")

rows = []
with open("$LOG") as f:
    for line in f:
        m = re.search(regex, line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            tok_s = float(m.group(3)) if m.lastindex and m.group(3) else float("nan")
            rows.append({"step": step, "loss": loss, "tok_per_s": tok_s})

with open("$METRICS", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["step","loss","tok_per_s"])
    w.writeheader()
    w.writerows(rows)

if rows:
    last = rows[-1]
    print(f"  {len(rows)} step records; final: step={last['step']} loss={last['loss']:.4f} "
          f"tok/s={last['tok_per_s']:.0f}")
else:
    print("  WARNING: no step lines matched the regex. Set OLMO_LOG_REGEX env to override.")
EOF
