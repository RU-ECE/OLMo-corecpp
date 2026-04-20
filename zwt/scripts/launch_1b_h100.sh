#!/usr/bin/env bash
# launch_1b_h100.sh — train the 1B zwt model on an H100, using pre-tokenized
# data that already lives on the attached volume so root disk stays clean.
#
# Assumptions:
#   * You're in the repo root (has zwt/conf/owt_1B_h100.conf).
#   * The tokenized OpenWebText .npy is already somewhere under
#     /media/volume/Prep_and_Voice_Training.
#   * An H100 is visible to nvidia-smi.
#
# What the script does:
#   1. Locates the tokenized .npy under the volume.
#   2. Symlinks it to data/owt/owt_tokens.npy (what the config expects).
#   3. Points ckpts/ at /media/volume/Prep_and_Voice_Training/zwt_ckpts/
#      so the ~10 GB checkpoints don't land on a full root disk.
#   4. Runs a --dry-run to verify model build.
#   5. Launches zwt_pretrain inside a tmux session so an ssh disconnect
#      doesn't kill the run. Logs to logs/1b_<timestamp>.log.
#
# Usage:
#   bash zwt/scripts/launch_1b_h100.sh            # locate + train
#   bash zwt/scripts/launch_1b_h100.sh --dry      # stop after dry-run
#   bash zwt/scripts/launch_1b_h100.sh --attach   # reattach to running run
#
# Stop a run:  tmux kill-session -t zwt_1b

set -euo pipefail

VOLUME="/media/volume/Prep_and_Voice_Training"
CONF="zwt/conf/owt_1B_h100.conf"
BIN="./build/zwt_pretrain"
TMUX_SESSION="zwt_1b"

DO_DRY_ONLY=0
DO_ATTACH=0
for a in "$@"; do
  case "$a" in
    --dry)    DO_DRY_ONLY=1 ;;
    --attach) DO_ATTACH=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

if [[ "$DO_ATTACH" -eq 1 ]]; then
  exec tmux attach -t "$TMUX_SESSION"
fi

# ---- 0. sanity: binary, config, GPU ----------------------------------------
if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — build first: bash ./scripts/build.sh" >&2
  exit 1
fi
if [[ ! -f "$CONF" ]]; then
  echo "missing $CONF — are you in the repo root?" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found — this is a GPU script, bailing" >&2
  exit 1
fi

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
echo

# ---- 1. locate the tokenized .npy on the volume ----------------------------
if [[ ! -d "$VOLUME" ]]; then
  echo "volume $VOLUME not mounted" >&2
  exit 1
fi

echo "=== Searching volume for tokenized data ==="
# Prefer a file whose name contains 'owt' or 'openwebtext' or 'tokens'.
CANDIDATES=$(find "$VOLUME" -maxdepth 6 -type f -name '*.npy' 2>/dev/null \
  | awk 'tolower($0) ~ /(owt|openwebtext|tokens)/' || true)
# Fallback: any sufficiently large .npy (>1 GB).
if [[ -z "$CANDIDATES" ]]; then
  CANDIDATES=$(find "$VOLUME" -maxdepth 6 -type f -name '*.npy' -size +1G 2>/dev/null || true)
fi

if [[ -z "$CANDIDATES" ]]; then
  echo "no .npy found under $VOLUME — pass the path explicitly:" >&2
  echo "  export ZWT_TOKENS=/media/volume/.../your_tokens.npy" >&2
  echo "  bash $0" >&2
  exit 1
fi

# If the user forced a path, use it. Otherwise pick the largest candidate.
if [[ -n "${ZWT_TOKENS:-}" ]]; then
  TOKENS="$ZWT_TOKENS"
else
  TOKENS=$(echo "$CANDIDATES" \
    | xargs -I{} stat -c '%s %n' {} \
    | sort -n | tail -1 | cut -d' ' -f2-)
fi

if [[ ! -f "$TOKENS" ]]; then
  echo "tokens file $TOKENS does not exist" >&2
  exit 1
fi
SIZE_GB=$(du -BG "$TOKENS" | cut -f1)
echo "using tokens: $TOKENS ($SIZE_GB)"
echo

# ---- 2. wire the data path to what the config expects ----------------------
mkdir -p data/owt
LINK="data/owt/owt_tokens.npy"
if [[ -L "$LINK" || -f "$LINK" ]]; then
  # Only replace if the link doesn't already point at the right file.
  EXISTING=$(readlink -f "$LINK" 2>/dev/null || echo "")
  if [[ "$EXISTING" != "$(readlink -f "$TOKENS")" ]]; then
    rm -f "$LINK"
    ln -s "$TOKENS" "$LINK"
  fi
else
  ln -s "$TOKENS" "$LINK"
fi
echo "symlink: $LINK -> $(readlink -f "$LINK")"

# ---- 3. checkpoints on the volume, not on /-----------------------------
CKPT_DIR_VOL="$VOLUME/zwt_ckpts"
mkdir -p "$CKPT_DIR_VOL"
if [[ -e ckpts && ! -L ckpts ]]; then
  echo "ckpts/ already exists as a real dir — leaving it alone." >&2
  echo "move it first if you want checkpoints on the volume:" >&2
  echo "  mv ckpts $CKPT_DIR_VOL/local && ln -s $CKPT_DIR_VOL ckpts" >&2
else
  rm -f ckpts
  ln -s "$CKPT_DIR_VOL" ckpts
fi
echo "ckpts dir: $(readlink -f ckpts)"
echo

# ---- 4. dry run: build the model, exit before the step loop ----------------
echo "=== Dry-run (build model, exit before step 1) ==="
"$BIN" "$CONF" --dry-run

if [[ "$DO_DRY_ONLY" -eq 1 ]]; then
  echo "dry-run requested; stopping."
  exit 0
fi

# ---- 5. launch under tmux ---------------------------------------------------
mkdir -p logs
LOG="logs/1b_$(date +%Y%m%d_%H%M%S).log"
echo "=== Launching training ==="
echo "session: $TMUX_SESSION"
echo "log:     $LOG"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session '$TMUX_SESSION' already exists — attach with:" >&2
  echo "  tmux attach -t $TMUX_SESSION" >&2
  echo "or kill it first:  tmux kill-session -t $TMUX_SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$TMUX_SESSION" \
  "stdbuf -oL -eL $BIN $CONF 2>&1 | tee $LOG"

echo
echo "training started in tmux."
echo "follow:   tmux attach -t $TMUX_SESSION      (detach with Ctrl-b then d)"
echo "tail:     tail -f $LOG"
echo "kill:     tmux kill-session -t $TMUX_SESSION"
