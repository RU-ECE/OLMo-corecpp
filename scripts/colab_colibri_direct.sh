#!/usr/bin/env bash
# colibri with DIRECT=1 (O_DIRECT expert reads: no page cache) on the Colab 2 vCPU / 12 GB CPU runtime, so the
# RAM it reports is the RAM it uses. Follow-up to colab_colibri_baseline.sh, whose RSS figures left ~9 GB of
# page cache uncounted. Prints everything to stdout (no Drive needed); results also in /content/moe/results.
set -uo pipefail
W=/content/moe; R=$W/results; mkdir -p "$R"; GG=Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
[ -d "$W/colibri" ] || git clone -q https://github.com/JustVugg/colibri "$W/colibri"
git -C "$W/colibri" checkout -q 0a5504a1 2>/dev/null; echo "colibri $(git -C "$W/colibri" rev-parse --short HEAD)"
[ -s "$W/$GG" ] || python3 -c "from huggingface_hub import hf_hub_download as d; d('unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', '$GG', local_dir='$W')"
cd "$W/colibri/c"
[ -d "$W/qwen3coder" ] || python3 ./coli import "$W/$GG" --out "$W/qwen3coder" > "$R/import.log" 2>&1
[ -x ./qwen36 ] || python3 ./coli build --model "$W/qwen3coder" > "$R/build.log" 2>&1
( while sleep 5; do free -m | awk '/Mem/{print strftime("%T"), "used", $3, "buff/cache", $6, "available", $7}'; done ) > "$R/mem_direct.log" &
MON=$!
for cap in ${CAPS:-8 32}; do
  sync; python3 -c "import os; fd=os.open('$W/$GG', os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)"; free -m | head -2
  DIRECT=1 timeout 1800 python3 tools/datapoint.py --snap "$W/qwen3coder" --engine ./qwen36 --cap "$cap" --memory-gb 10 \
    --max-new 64 --warm-runs 1 --rotating-runs 2 --no-evict > "$R/direct_cap$cap.txt" 2>&1
  echo "=== DIRECT=1 cap=$cap exit $?"; grep -aE "^\| (cold|warm)" "$R/direct_cap$cap.txt"
  grep -aE "expert disk|expert matmul|attention" "$R/direct_cap$cap.txt" | tail -3
done
kill $MON
echo "=== memory while running (MB): max used, max buff/cache"
awk '{u=$3>u?$3:u; c=$5>c?$5:c} END {print "used", u, "buff/cache", c}' "$R/mem_direct.log"
echo DIRECT_DONE
