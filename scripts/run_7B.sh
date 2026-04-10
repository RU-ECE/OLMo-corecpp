#!/usr/bin/env bash
# run_7B.sh — Launch 7B training in tmux with heartbeat monitoring + email alerts
#
# Usage:
#   ./scripts/run_7B.sh --emails "you@example.com,prof@example.com"
#   ./scripts/run_7B.sh --emails "you@example.com" --discord "https://discord.com/api/webhooks/..."
#
# Prerequisites:
#   sudo apt-get install -y mailutils msmtp msmtp-mta
#   # Configure ~/.msmtprc with your SMTP relay (e.g. Gmail app password)

set -euo pipefail

# ── Defaults ──
CONF="conf/olmo_7B_h100.conf"
LOG="7B.log"
HEARTBEAT="heartbeat_7B.txt"
STALE_TIMEOUT=600           # seconds without heartbeat update before alerting
POLL_INTERVAL=60            # how often the monitor checks the heartbeat file
SESSION="train7B"
EMAILS=""                   # comma-separated list of emails to alert
DISCORD_WEBHOOK=""          # Discord webhook URL (optional)

# ── Parse args ──
while [[ $# -gt 0 ]]; do
  case "$1" in
    --conf)       CONF="$2";           shift 2 ;;
    --log)        LOG="$2";            shift 2 ;;
    --heartbeat)  HEARTBEAT="$2";      shift 2 ;;
    --timeout)    STALE_TIMEOUT="$2";  shift 2 ;;
    --emails)     EMAILS="$2";         shift 2 ;;
    --discord)    DISCORD_WEBHOOK="$2"; shift 2 ;;
    --session)    SESSION="$2";        shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$EMAILS" && -z "$DISCORD_WEBHOOK" ]]; then
  echo "ERROR: --emails and/or --discord is required"
  echo "  e.g.: ./scripts/run_7B.sh --emails 'you@uni.edu' --discord 'https://discord.com/api/webhooks/...'"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Verify binary exists
if [[ ! -x build/olmo_train ]]; then
  echo "ERROR: build/olmo_train not found. Run ./scripts/build.sh --cuda first."
  exit 1
fi

# Verify config exists
if [[ ! -f "$CONF" ]]; then
  echo "ERROR: Config file $CONF not found."
  exit 1
fi

HOSTNAME="$(hostname)"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

# ── Alert helpers ──
send_email() {
  local subject="$1"
  local body="$2"
  [[ -z "$EMAILS" ]] && return
  IFS=',' read -ra ADDR <<< "$EMAILS"
  for addr in "${ADDR[@]}"; do
    addr="$(echo "$addr" | xargs)"  # trim whitespace
    echo "$body" | mail -s "$subject" "$addr" 2>/dev/null || \
      echo "[$(date)] WARNING: Failed to send email to $addr"
  done
}

send_discord() {
  local subject="$1"
  local body="$2"
  [[ -z "$DISCORD_WEBHOOK" ]] && return
  # Discord embeds: color red=16711680, green=65280, yellow=16776960
  local color=16776960  # yellow default
  if [[ "$subject" == *"FAILED"* || "$subject" == *"STALLED"* ]]; then
    color=16711680  # red
  elif [[ "$subject" == *"COMPLETED"* || "$subject" == *"RECOVERED"* ]]; then
    color=65280  # green
  elif [[ "$subject" == *"Started"* ]]; then
    color=3447003  # blue
  fi
  # Truncate body to 4000 chars (Discord limit is 4096)
  local desc="${body:0:4000}"
  # Escape for JSON
  desc=$(echo "$desc" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
  local payload="{\"embeds\":[{\"title\":$(echo "$subject" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))'),\"description\":$desc,\"color\":$color}]}"
  curl -s -H "Content-Type: application/json" -d "$payload" "$DISCORD_WEBHOOK" >/dev/null 2>&1 || \
    echo "[$(date)] WARNING: Failed to send Discord notification"
}

send_alert() {
  send_email "$1" "$2"
  send_discord "$1" "$2"
}

# ── Clean up any old heartbeat file ──
rm -f "$HEARTBEAT"

echo "============================================"
echo "  7B Training Launch"
echo "============================================"
echo "  Config:     $CONF"
echo "  Log:        $LOG"
echo "  Heartbeat:  $HEARTBEAT"
echo "  Timeout:    ${STALE_TIMEOUT}s"
echo "  Emails:     ${EMAILS:-none}"
echo "  Discord:    ${DISCORD_WEBHOOK:+enabled}"
echo "  tmux:       $SESSION"
echo "============================================"

# ── Launch training in a tmux session ──
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" \
  "cd $PROJECT_DIR && ./build/olmo_train $CONF >$LOG 2>&1; echo EXIT_CODE=\$? >> $LOG"

echo "[$(date)] Training started in tmux session '$SESSION'"
echo "  Attach:  tmux attach -t $SESSION"
echo "  Log:     tail -f $LOG"

# Send start notification
send_alert \
  "[7B Training] Started on $HOSTNAME" \
  "Training job started at $TIMESTAMP on $HOSTNAME
Config: $CONF
Log: $PROJECT_DIR/$LOG
tmux session: $SESSION

To attach: ssh $HOSTNAME -t 'tmux attach -t $SESSION'"

# ── Monitor loop ──
echo "[$(date)] Monitor started (timeout=${STALE_TIMEOUT}s, poll=${POLL_INTERVAL}s)"

alerted=0
while true; do
  sleep "$POLL_INTERVAL"

  # Check if training process is still running
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    # Training ended — check exit status
    exit_line=$(grep 'EXIT_CODE=' "$LOG" 2>/dev/null | tail -1 || true)
    exit_code="${exit_line#EXIT_CODE=}"
    last_epoch=$(grep -oP 'Epoch \K[0-9]+' "$LOG" 2>/dev/null | tail -1 || echo "?")
    last_loss=$(grep -oP 'loss: \K[0-9.]+' "$LOG" 2>/dev/null | tail -1 || echo "?")
    tail_log=$(tail -30 "$LOG" 2>/dev/null || echo "(no log)")

    if [[ "$exit_code" == "0" ]]; then
      send_alert \
        "[7B Training] COMPLETED on $HOSTNAME" \
        "Training completed successfully at $(date).

Last epoch: $last_epoch
Last loss:  $last_loss

Last 30 lines of log:
$tail_log"
      echo "[$(date)] Training completed successfully."
    else
      send_alert \
        "[7B Training] FAILED (exit $exit_code) on $HOSTNAME" \
        "Training FAILED at $(date) with exit code: $exit_code

Last epoch: $last_epoch
Last loss:  $last_loss

Last 30 lines of log:
$tail_log"
      echo "[$(date)] Training FAILED with exit code: $exit_code"
    fi
    break
  fi

  # Check heartbeat freshness
  if [[ -f "$HEARTBEAT" ]]; then
    last_mod=$(stat -c %Y "$HEARTBEAT" 2>/dev/null || stat -f %m "$HEARTBEAT" 2>/dev/null)
    now=$(date +%s)
    age=$(( now - last_mod ))

    if (( age > STALE_TIMEOUT )); then
      if (( alerted == 0 )); then
        last_epoch=$(grep -oP 'Epoch \K[0-9]+' "$LOG" 2>/dev/null | tail -1 || echo "?")
        last_loss=$(grep -oP 'loss: \K[0-9.]+' "$LOG" 2>/dev/null | tail -1 || echo "?")
        hb_content=$(cat "$HEARTBEAT" 2>/dev/null || echo "(empty)")
        tail_log=$(tail -20 "$LOG" 2>/dev/null || echo "(no log)")

        send_alert \
          "[7B Training] STALLED on $HOSTNAME - no heartbeat for ${age}s" \
          "WARNING: Training appears stalled on $HOSTNAME at $(date).
Heartbeat file not updated for ${age} seconds (threshold: ${STALE_TIMEOUT}s).

Heartbeat contents:
$hb_content

Last epoch: $last_epoch
Last loss:  $last_loss

Last 20 lines of log:
$tail_log

Action needed: ssh $HOSTNAME and check tmux attach -t $SESSION"

        echo "[$(date)] ALERT: Heartbeat stale for ${age}s — emails sent."
        alerted=1
      fi
    else
      if (( alerted == 1 )); then
        echo "[$(date)] Heartbeat recovered (age=${age}s)."
        send_alert \
          "[7B Training] RECOVERED on $HOSTNAME" \
          "Training heartbeat recovered at $(date). Age: ${age}s."
      fi
      alerted=0
      # Print status
      hb_epoch=$(grep 'epoch=' "$HEARTBEAT" 2>/dev/null | head -1 | cut -d= -f2 || echo "?")
      hb_step=$(grep 'step=' "$HEARTBEAT" 2>/dev/null | head -1 | cut -d= -f2 || echo "?")
      echo "[$(date)] OK: epoch=$hb_epoch step=$hb_step (heartbeat age=${age}s)"
    fi
  else
    # No heartbeat file yet — training is still in warmup/first epochs
    echo "[$(date)] Waiting for heartbeat file..."
  fi
done

echo "[$(date)] Monitor exiting."
