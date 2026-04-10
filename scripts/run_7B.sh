#!/usr/bin/env bash
# run_7B.sh — Launch 7B training in tmux with monitoring
#
# Reads .env.alerts for secrets (emails, Discord webhooks).
# Two Discord channels:
#   DISCORD_ALERTS  — fail/stall/complete/start (critical events only)
#   DISCORD_UPDATES — progress every UPDATE_INTERVAL seconds (~10 min)
#
# Usage:
#   ./scripts/run_7B.sh
#   ./scripts/run_7B.sh --conf conf/olmo_7B_h100.conf --timeout 600

set -euo pipefail

# ── Defaults ──
CONF="conf/olmo_7B_h100.conf"
LOG="7B.log"
HEARTBEAT="heartbeat_7B.txt"
STALE_TIMEOUT=600
POLL_INTERVAL=30
UPDATE_INTERVAL=600         # Discord progress update interval (seconds)
SESSION="train7B"
ENV_FILE=".env.alerts"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
  case "$1" in
    --conf)       CONF="$2";            shift 2 ;;
    --log)        LOG="$2";             shift 2 ;;
    --heartbeat)  HEARTBEAT="$2";       shift 2 ;;
    --timeout)    STALE_TIMEOUT="$2";   shift 2 ;;
    --update)     UPDATE_INTERVAL="$2"; shift 2 ;;
    --session)    SESSION="$2";         shift 2 ;;
    --env)        ENV_FILE="$2";        shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ── Load secrets from env file ──
ALERT_EMAILS=""
DISCORD_ALERTS=""
DISCORD_UPDATES=""

if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r key val; do
    key=$(echo "$key" | xargs)
    [[ -z "$key" || "$key" == \#* ]] && continue
    val=$(echo "$val" | sed 's/^["'\'']*//;s/["'\'']*$//')
    case "$key" in
      ALERT_EMAILS)    ALERT_EMAILS="$val" ;;
      DISCORD_ALERTS)  DISCORD_ALERTS="$val" ;;
      DISCORD_UPDATES) DISCORD_UPDATES="$val" ;;
    esac
  done < "$ENV_FILE"
  echo "Loaded config from $ENV_FILE"
else
  echo "WARNING: $ENV_FILE not found. cp .env.alerts.example .env.alerts"
fi

# ── Verify prerequisites ──
if [[ ! -x build/olmo_train ]]; then
  echo "ERROR: build/olmo_train not found. Run ./scripts/build.sh --cuda first."
  exit 1
fi
if [[ ! -f "$CONF" ]]; then
  echo "ERROR: Config $CONF not found."
  exit 1
fi

HOSTNAME="$(hostname)"

# ── Messaging helpers ──
send_email() {
  local subject="$1" body="$2"
  [[ -z "$ALERT_EMAILS" ]] && return
  IFS=',' read -ra ADDR <<< "$ALERT_EMAILS"
  for addr in "${ADDR[@]}"; do
    addr="$(echo "$addr" | xargs)"
    [[ -z "$addr" ]] && continue
    echo "$body" | mail -s "$subject" "$addr" 2>/dev/null || \
      echo "[$(date)] WARNING: email to $addr failed"
  done
}

_discord_post() {
  local webhook="$1" subject="$2" body="$3" color="$4"
  [[ -z "$webhook" ]] && return
  local desc="${body:0:4000}"
  desc=$(printf '%s' "$desc" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
  local title
  title=$(printf '%s' "$subject" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')
  local payload="{\"embeds\":[{\"title\":$title,\"description\":$desc,\"color\":$color}]}"
  curl -s -H "Content-Type: application/json" -d "$payload" "$webhook" >/dev/null 2>&1 || \
    echo "[$(date)] WARNING: Discord post failed"
}

# Critical alerts → email + alerts channel
send_alert() {
  local subject="$1" body="$2"
  local color=16776960  # yellow
  if [[ "$subject" == *"FAILED"* || "$subject" == *"STALLED"* ]]; then color=16711680; fi  # red
  if [[ "$subject" == *"COMPLETED"* || "$subject" == *"RECOVERED"* ]]; then color=65280; fi  # green
  if [[ "$subject" == *"Started"* ]]; then color=3447003; fi  # blue

  send_email "$subject" "$body"
  _discord_post "$DISCORD_ALERTS" "$subject" "$body" "$color"
}

# Progress → updates channel only
send_progress() {
  local subject="$1" body="$2"
  _discord_post "$DISCORD_UPDATES" "$subject" "$body" 3447003  # blue
}

# ── Get latest training stats from log ──
get_stats() {
  local last_line
  last_line=$(grep -E 'Step [0-9]+/' "$LOG" 2>/dev/null | tail -1 || echo "")
  if [[ -z "$last_line" ]]; then
    echo "epoch=? step=? loss=? tok/s=?"
    return
  fi
  local epoch step loss toks
  epoch=$(echo "$last_line" | grep -oP 'Epoch \K[0-9]+' || echo "?")
  step=$(echo "$last_line" | grep -oP 'Step \K[0-9]+/[0-9]+' || echo "?")
  loss=$(echo "$last_line" | grep -oP 'loss: \K[0-9.]+' || echo "?")
  toks=$(echo "$last_line" | grep -oP 'tok/s: \K[0-9]+' || echo "?")
  echo "epoch=$epoch step=$step loss=$loss tok/s=$toks"
}

# ── Clean up ──
rm -f "$HEARTBEAT"

echo "============================================"
echo "  7B Training Launch"
echo "============================================"
echo "  Config:      $CONF"
echo "  Log:         $LOG"
echo "  Heartbeat:   $HEARTBEAT"
echo "  Stale:       ${STALE_TIMEOUT}s"
echo "  Updates:     every ${UPDATE_INTERVAL}s"
echo "  Emails:      ${ALERT_EMAILS:-none}"
echo "  Discord:     alerts=${DISCORD_ALERTS:+yes} updates=${DISCORD_UPDATES:+yes}"
echo "  tmux:        $SESSION"
echo "============================================"

# ── Launch training ──
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd $PROJECT_DIR && ./build/olmo_train $CONF >$LOG 2>&1; echo EXIT_CODE=\$? >> $LOG"

echo "[$(date)] Training started in tmux '$SESSION'"
echo "  Attach: tmux attach -t $SESSION"
echo "  Log:    tail -f $LOG"

send_alert "[7B] Started on $HOSTNAME" \
  "Config: $CONF
Host: $HOSTNAME
Time: $(date)
tmux: $SESSION"

# ── Monitor loop ──
alerted=0
last_update=0

while true; do
  sleep "$POLL_INTERVAL"
  now=$(date +%s)

  # ── Check if training ended ──
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    exit_line=$(grep 'EXIT_CODE=' "$LOG" 2>/dev/null | tail -1 || true)
    exit_code="${exit_line#EXIT_CODE=}"
    stats=$(get_stats)
    tail_log=$(tail -30 "$LOG" 2>/dev/null || echo "(no log)")

    if [[ "$exit_code" == "0" ]]; then
      send_alert "[7B] COMPLETED on $HOSTNAME" \
        "Training completed at $(date).
$stats

$tail_log"
      send_progress "[7B] COMPLETED" "$stats"
      echo "[$(date)] Training completed."
    else
      send_alert "[7B] FAILED (exit $exit_code) on $HOSTNAME" \
        "FAILED at $(date) with exit code: $exit_code
$stats

$tail_log"
      send_progress "[7B] FAILED (exit $exit_code)" "$stats"
      echo "[$(date)] Training FAILED: exit $exit_code"
    fi
    break
  fi

  # ── Send periodic progress to Discord updates channel ──
  if (( now - last_update >= UPDATE_INTERVAL )); then
    stats=$(get_stats)
    if [[ "$stats" != *"epoch=?"* ]]; then
      send_progress "[7B] Progress on $HOSTNAME" "$stats"
      echo "[$(date)] Progress: $stats"
    fi
    last_update=$now
  fi

  # ── Check heartbeat staleness ──
  if [[ -f "$HEARTBEAT" ]]; then
    last_mod=$(stat -c %Y "$HEARTBEAT" 2>/dev/null || stat -f %m "$HEARTBEAT" 2>/dev/null)
    age=$(( now - last_mod ))

    if (( age > STALE_TIMEOUT )); then
      if (( alerted == 0 )); then
        stats=$(get_stats)
        hb=$(cat "$HEARTBEAT" 2>/dev/null || echo "(empty)")
        tail_log=$(tail -20 "$LOG" 2>/dev/null || echo "(no log)")
        send_alert "[7B] STALLED on $HOSTNAME (${age}s)" \
          "No heartbeat for ${age}s (threshold: ${STALE_TIMEOUT}s)
$stats

Heartbeat: $hb

$tail_log

ssh $HOSTNAME -t 'tmux attach -t $SESSION'"
        echo "[$(date)] ALERT: heartbeat stale ${age}s"
        alerted=1
      fi
    else
      if (( alerted == 1 )); then
        send_alert "[7B] RECOVERED on $HOSTNAME" "Heartbeat recovered. Age: ${age}s"
        echo "[$(date)] Recovered."
      fi
      alerted=0
    fi
  fi
done

echo "[$(date)] Monitor exiting."
