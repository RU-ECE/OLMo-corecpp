#!/usr/bin/env bash
# Big-MoE-on-a-small-machine baseline: colibri on Qwen3-Coder-30B-A3B (Q4_K_M GGUF, 18.6 GB) on a
# Colab CPU runtime (2 vCPU, 12 GB RAM: the model does not fit in RAM), with llama.cpp on the same file
# as the control. This is the number our own expert-streaming engine has to beat (plan: Gaps to close, row 6).
#
#   bash colab_colibri_baseline.sh           (Colab CPU runtime; nothing else may run on it meanwhile)
#
# colibri is measured with its own tools/datapoint.py (persistent mode: page cache evicted with
# posix_fadvise first, then one cold request, one warm repeat, rotating prompts; the engine's DONE frame
# gives decode tok/s, expert hit rate and RSS). One variable at a time: the per-layer expert cache cap.
# `coli run`'s resource planner cannot read Colab's cgroup path, so the engine is driven directly and the
# RAM budget is given explicitly (--memory-gb). Every number is labelled 2 vCPU / 12 GB.
set -uo pipefail
W=${WORK:-/content/moe}; R=$W/results; SAVE=${SAVE:-/content/drive/MyDrive/llm-cpp-moe-baseline}
NGEN=${NGEN:-64}; CAPS=${CAPS:-"32 16 8"}; MEM=${MEM:-10}; LLAMA_MIN=${LLAMA_MIN:-20}
mkdir -p "$W" "$R"; [ "$SAVE" != none ] && [ -d "$(dirname "$SAVE")" ] && mkdir -p "$SAVE" || SAVE=none
save() { [ "$SAVE" != none ] && cp -f "$@" "$SAVE"/ 2>/dev/null; true; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$R/run.log"; save "$R/run.log"; }
export CUDA_VISIBLE_DEVICES=""
GG=Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
{ lscpu | grep -E "Model name|^CPU\(s\)"; free -g; df -h "$W" | tail -1; } > "$R/machine.txt" 2>&1
log "machine: $(grep 'Model name' "$R/machine.txt" | sed 's/.*: *//'), $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GB RAM"

[ -d "$W/colibri" ] || git clone -q https://github.com/JustVugg/colibri "$W/colibri"
log "colibri $(git -C "$W/colibri" rev-parse --short HEAD)"
[ -s "$W/$GG" ] || python3 -c "from huggingface_hub import hf_hub_download as d; d('unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', '$GG', local_dir='$W')" \
  || { log "FAILED download"; exit 1; }
C=$W/colibri/c; cd "$C"
[ -d "$W/qwen3coder" ] || python3 ./coli import "$W/$GG" --out "$W/qwen3coder" > "$R/import.log" 2>&1 || { log "FAILED import"; exit 1; }
[ -x "$C/qwen36" ] || python3 ./coli build --model "$W/qwen3coder" > "$R/build.log" 2>&1 || { log "FAILED build"; exit 1; }

for cap in $CAPS; do
  out="$R/datapoint_cap${cap}.txt"
  log "colibri datapoint cap=$cap memory=${MEM}GB max-new=$NGEN"
  timeout 3600 python3 tools/datapoint.py --snap "$W/qwen3coder" --engine ./qwen36 --cap "$cap" --memory-gb "$MEM" \
    --max-new "$NGEN" --warm-runs 1 --rotating-runs 2 > "$out" 2>&1
  log "  exit $?; $(grep -aiE 'decode|tok/s|hit|RSS' "$out" | tail -4 | tr '\n' ' ' | cut -c1-400)"
  save "$out"
done

log "control: llama.cpp b11900 CPU (mmap, 2 threads) on the same GGUF, capped at ${LLAMA_MIN} min"
L=$W/llama.cpp
if [ ! -x "$L/build/bin/llama-bench" ]; then
  [ -d "$L" ] || git clone -q --depth 1 --branch b11900 https://github.com/ggml-org/llama.cpp "$L"
  cmake -S "$L" -B "$L/build" -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$R/llama_build.log" 2>&1 \
    && cmake --build "$L/build" -j"$(nproc)" --target llama-bench >> "$R/llama_build.log" 2>&1 || log "FAILED llama.cpp build"
fi
sync; python3 -c "import os; fd=os.open('$W/$GG', os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)"
timeout $((LLAMA_MIN * 60)) "$L/build/bin/llama-bench" -m "$W/$GG" -t "$(nproc)" -p 32 -n "$NGEN" -r 1 -o md > "$R/llama_bench.md" 2>&1
rc=$?; [ $rc = 124 ] && echo "did not finish in ${LLAMA_MIN} min" >> "$R/llama_bench.md"
log "llama.cpp exit $rc"; cat "$R/llama_bench.md" | tee -a "$R/run.log"; save "$R"/*
log "MOE_BASELINE_DONE"
