#!/usr/bin/env bash
# Knowsy Forge demo on one Colab A100: DoRA SFT of OLMo-2-1124-7B on Forge agent
# trajectories with every applicable speedup, plus the ablation that measures each.
#
#   unzip llm-cpp.zip -d /content && unzip forge_data.zip -d /content
#   bash /content/llm-cpp/scripts/colab_forge_demo.sh            (all stages)
#   STAGES="data model ablation train" bash ...                   (subset)
#
# Stages
#   build     olmo_train, convert_hf, bench_spec_decode (CUDA, this GPU's arch)
#   data      trajectories -> sft.jsonl (gate-passing runs) + a Tulu-3 slice for
#             general skill -> packed tokens + assistant-only loss mask
#   model     OLMo-2-1124-7B from Hugging Face -> llm-cpp checkpoint
#   ablation  same data, same steps, each config timed (steady-state tok/s from
#             step_ms, peak GPU memory from nvidia-smi, final loss):
#               baseline   materialised DoRA + full-rank MTP heads (today's recipe)
#               +fdora     factored DoRA
#               +slmtp     factored DoRA + SL-MTP heads (chunked loss, CUDA row kernels)
#   train     the demo model: all speedups, EPOCHS over the data
#   export    (opt-in) merge adapters -> HF Olmo2 -> GGUF f16 -> Q4_K_M for the Forge app
# Everything lands in $VOL/results (results.json + one log per run).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOL=${VOL:-/content/forge_demo}
FORGE=${FORGE:-/content/forge_data}
STAGES=${STAGES:-"build data model ablation train"}
# 4096 = OLMo-2-1124-7B's trained context, which is also the cap Forge's qrt runtime
# applies (n_ctx <= n_ctx_train). Forge trajectories are 3k-12k tokens: at 2048 every
# one is dropped; at 4096, 7 of 12 fit (measured 2026-11-14).
SEQ=${SEQ:-4096}
ABL_STEPS=${ABL_STEPS:-8}
ABL_ACCUM=${ABL_ACCUM:-4}
EPOCHS=${EPOCHS:-3}
TRAIN_ACCUM=${TRAIN_ACCUM:-4}
R=$VOL/results
mkdir -p "$R" "$VOL/runs"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$R/demo.log"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
# Fill ${VAR}s in a config template from the environment. Python, not envsubst: Colab
# has no gettext, and envsubst left every config empty there. An unset variable is an error.
render() {
  python3 -c 'import os, string, sys; sys.stdout.write(string.Template(open(sys.argv[1]).read()).substitute(os.environ))' \
    "$1" > "$2" && grep -q '^\[model\]' "$2" || { log "config render FAILED: $1 -> $2"; exit 1; }
}

if has build; then
  log "build"
  pip -q install ninja >/dev/null 2>&1 || true
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
  INJECT=""; EXTRA=""
  if [ -f "$FORGE/training/llmcpp_inject.cmake" ]; then  # Forge's Olmo2 exporter, for the export stage
    INJECT="-DCMAKE_PROJECT_INCLUDE=$FORGE/training/llmcpp_inject.cmake"; EXTRA=export_hf_olmo2
  fi
  cmake -G Ninja -S "$ROOT" -B "$ROOT/build-cuda" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DOLMO_RACE_BUILD=OFF $INJECT >> "$R/build.log" 2>&1 || { log "configure FAILED"; exit 1; }
  ninja -C "$ROOT/build-cuda" -j"$(nproc)" olmo_train convert_hf bench_spec_decode $EXTRA >> "$R/build.log" 2>&1 \
    || { log "build FAILED"; grep -n error "$R/build.log" | head; exit 1; }
fi
B=$ROOT/build-cuda

if has data; then
  log "data"
  pip -q install tokenizers datasets huggingface_hub >/dev/null 2>&1 || true
  mkdir -p "$VOL/olmo2-7b" "$VOL/forge-data"
  python - <<PY
from huggingface_hub import hf_hub_download
for f in ["tokenizer.json", "vocab.json", "merges.txt", "config.json"]:
    hf_hub_download("allenai/OLMo-2-1124-7B", f, local_dir="$VOL/olmo2-7b")
PY
  python -I "$FORGE/export_sft.py" "$FORGE/trajectories" "$VOL/forge-data" | tee -a "$R/demo.log"
  # General instruction data so a small agent set does not erase general skill.
  if [ "$(wc -l < "$VOL/forge-data/tulu.jsonl" 2>/dev/null || echo 0)" -ge 400 ]; then
    echo "tulu.jsonl already has 400 conversations; reusing it"
  else
  python - <<PY
import json, os, sys
from datasets import load_dataset
ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
with open("$VOL/forge-data/tulu.jsonl", "w") as f:
    n = 0
    for ex in ds:
        msgs = [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
        if sum(len(m["content"]) for m in msgs) > 6000: continue
        f.write(json.dumps({"messages": msgs}) + "\n"); n += 1
        if n >= 400: break
print("tulu conversations:", n)
sys.stdout.flush()
os._exit(0)  # the streaming reader's worker threads can hang interpreter exit
PY
  fi
  python "$FORGE/prep_forge_sft.py" --tokenizer "$VOL/olmo2-7b/tokenizer.json" \
    --forge "$VOL/forge-data/sft.jsonl" --general "$VOL/forge-data/tulu.jsonl" --general-ratio 3 \
    --repeat-forge 4 --seq-len "$SEQ" --out "$VOL/forge-sft" | tee "$R/data.txt"
  if grep -q "kept forge=0 " "$R/data.txt"; then
    log "data FAILED: no Forge conversation fits in SEQ=$SEQ tokens (see $R/data.txt); the demo would learn nothing from Forge"
    exit 1
  fi
  # Every conversation must sit whole inside one training window (prep's --layout windows).
  if [ -f "$FORGE/check_sft_windows.py" ]; then
    python "$FORGE/check_sft_windows.py" "$VOL/olmo2-7b/tokenizer.json" "$SEQ" "$VOL/forge-sft" | tee -a "$R/data.txt" \
      || { log "data FAILED: a conversation straddles a training window (see $R/data.txt)"; exit 1; }
  fi
fi

if has model; then
  log "model: download + convert OLMo-2-1124-7B"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("allenai/OLMo-2-1124-7B", local_dir="$VOL/olmo2-7b",
                  allow_patterns=["*.safetensors", "*.json", "*.txt"])
PY
  "$B/convert_hf" --hf-dir "$VOL/olmo2-7b" --config "$ROOT/configs/olmo2_1124_7B.json" \
    --output "$VOL/olmo2-7b/olmo2_7b.pt" 2>&1 | tail -3 | tee -a "$R/demo.log"
  rm -f "$VOL"/olmo2-7b/*.safetensors   # the .pt has everything
fi

# One training run: $1 name, $2 steps, $3 accum, $4 mtp heads, $5 type, $6 rank, $7 log interval, rest = env
run() {
  local name=$1 steps=$2 accum=$3 k=$4 type=$5 rank=$6 logi=$7; shift 7
  export RUN=$name VOL STEPS=$steps SEQ ACCUM=$accum MTP_K=$k MTP_TYPE=$type MTP_RANK=$rank LOG=$logi
  export SAVE_LINE="# no checkpoint for timing runs" CKPT_INT=100000 RESUME=${RESUME:-0}
  # The demo run's last step leaves a resumable checkpoint; export merges from it, so no
  # separate full-model `save` (another 14.8 GB written and held in RAM at the end).
  [ "$name" = forge_demo ] && export CKPT_INT=$steps
  render "$ROOT/conf/forge_demo_7b.conf.in" "$VOL/runs/$name.conf"
  mkdir -p "$VOL/runs/$name"
  # Peak GPU memory: sample nvidia-smi every 0.5 s while the run lasts.
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) \
    > "$R/$name.mem" 2>/dev/null &
  local mon=$!
  local t0=$(date +%s)
  env "$@" "$B/olmo_train" "$VOL/runs/$name.conf" > "$R/$name.log" 2>&1
  local rc=$?
  kill $mon 2>/dev/null; wait $mon 2>/dev/null
  log "run $name exit $rc in $(( $(date +%s) - t0 )) s"
  [ $rc -eq 0 ] || tail -5 "$R/$name.log" | tee -a "$R/demo.log"
  return $rc
}

if has ablation; then
  log "ablation: $ABL_STEPS steps x accum $ABL_ACCUM x seq $SEQ"
  # A failed timing run is logged and the next one still runs.
  run baseline "$ABL_STEPS" "$ABL_ACCUM" 2 full 0   1 OLMO_DORA_MATERIALIZED=1
  run fdora    "$ABL_STEPS" "$ABL_ACCUM" 2 full 0   1
  run slmtp    "$ABL_STEPS" "$ABL_ACCUM" 3 shared_lowrank 128 1
fi

if has train; then
  NSEQ=$(python -c "import numpy as np; print(np.load('$VOL/forge-sft/sft_tokens.npy', mmap_mode='r').size // $SEQ)")
  STEPS=$(( (NSEQ * EPOCHS + TRAIN_ACCUM - 1) / TRAIN_ACCUM ))
  log "demo training: $NSEQ sequences x $EPOCHS epochs -> $STEPS steps (accum $TRAIN_ACCUM)"
  run forge_demo "$STEPS" "$TRAIN_ACCUM" 3 shared_lowrank 128 1 || { log "demo training FAILED; not exporting"; exit 1; }
fi

if has export; then
  # Demo model -> the Forge app: fold the adapters in (resume-load the final
  # checkpoint), HF Olmo2 (MTP heads are skipped: llama.cpp has no slot for them),
  # GGUF f16, then Q4_K_M for qrt. llama.cpp pinned to Forge's b11900.
  D=$VOL/runs/forge_demo
  log "export: merge"
  NSEQ=$(python -c "import numpy as np; print(np.load('$VOL/forge-sft/sft_tokens.npy', mmap_mode='r').size // $SEQ)")
  STEPS=$(( (NSEQ * EPOCHS + TRAIN_ACCUM - 1) / TRAIN_ACCUM ))
  export RUN=forge_demo VOL STEPS SEQ ACCUM=$TRAIN_ACCUM MTP_K=3 MTP_TYPE=shared_lowrank MTP_RANK=128 LOG=1 \
         SAVE_LINE="# merge only" CKPT_INT=$STEPS RESUME=1
  render "$ROOT/conf/forge_demo_7b.conf.in" "$D.merge.conf"
  MERGE_EXPORT="$D/merged.pt" "$B/olmo_train" "$D.merge.conf" > "$R/merge.log" 2>&1; rc=$?; log "merge exit $rc"
  [ $rc -eq 0 ] && [ -s "$D/merged.pt" ] || { tail -5 "$R/merge.log" | tee -a "$R/demo.log"; log "merge FAILED"; exit 1; }
  "$B/export_hf_olmo2" --config "$ROOT/configs/olmo2_1124_7B.json" --ckpt "$D/merged.pt" --out "$D/hf" \
    --tokenizer-dir "$VOL/olmo2-7b" 2>&1 | tail -2 | tee -a "$R/demo.log"
  [ ${PIPESTATUS[0]} -eq 0 ] || { log "HF export FAILED"; exit 1; }
  [ -d /content/llama.cpp ] || git clone -q --depth 1 --branch b11900 https://github.com/ggml-org/llama.cpp /content/llama.cpp \
    || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
  # NOT llama.cpp's requirements file: it pins a CPU-only torch (==2.11.0) and would replace the CUDA
  # torch that olmo_train links against. The converter needs only these (gguf comes from llama.cpp/gguf-py).
  pip -q install sentencepiece safetensors >/dev/null 2>&1
  python -c "import torch; assert torch.cuda.is_available(), torch.__version__" || { log "torch was changed; stopping"; exit 1; }
  python /content/llama.cpp/convert_hf_to_gguf.py "$D/hf" --outtype f16 --outfile "$D/forge-demo-olmo2-7b-f16.gguf" \
    > "$R/gguf.log" 2>&1; log "gguf exit $?"
  cmake -S /content/llama.cpp -B /content/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null 2>&1
  cmake --build /content/llama.cpp/build --target llama-quantize -j"$(nproc)" > /dev/null 2>&1
  /content/llama.cpp/build/bin/llama-quantize "$D/forge-demo-olmo2-7b-f16.gguf" "$D/forge-demo-olmo2-7b-q4_k_m.gguf" Q4_K_M \
    > "$R/quantize.log" 2>&1; log "quantize exit $?"
  ls -la "$D"/*.gguf 2>/dev/null | tee -a "$R/demo.log"
fi

# Results: one JSON for the paper / the app.
python - <<PY
import json, os, re, statistics, subprocess
R, VOL, SEQ = "$R", "$VOL", int("$SEQ")
def parse(name, accum):
    p = f"{R}/{name}.log"
    if not os.path.exists(p): return None
    txt = open(p).read()
    steps = [(int(m.group(1)), float(m.group(2)), int(m.group(3)))
             for m in re.finditer(r"Step (\d+)/\d+\s+loss: ([0-9.]+).*?step_ms: (\d+)", txt)]
    mem = [int(x) for x in open(f"{R}/{name}.mem").read().split() if x.isdigit()] if os.path.exists(f"{R}/{name}.mem") else []
    if not steps: return {"name": name, "error": txt[-1500:]}
    steady = [ms for (s, _, ms) in steps[2:]] or [ms for (_, _, ms) in steps]  # skip warm-up steps
    ms = statistics.median(steady)
    return {"name": name, "steps": len(steps), "first_loss": steps[0][1], "final_loss": steps[-1][1],
            "step_ms_median": ms, "tokens_per_step": SEQ * accum,
            "steady_tok_s": round(SEQ * accum / (ms / 1000.0), 1),
            "peak_gpu_mem_mib": max(mem) if mem else None}
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
res = {"gpu": gpu, "seq_len": SEQ, "data": open(f"{R}/data.txt").read() if os.path.exists(f"{R}/data.txt") else None,
       "ablation": [r for r in (parse(n, int("$ABL_ACCUM")) for n in ["baseline", "fdora", "slmtp"]) if r],
       "demo": parse("forge_demo", int("$TRAIN_ACCUM"))}
base = next((r for r in res["ablation"] if r["name"] == "baseline" and "steady_tok_s" in r), None)
for r in res["ablation"]:
    if base and "steady_tok_s" in r:
        r["speedup_vs_baseline"] = round(r["steady_tok_s"] / base["steady_tok_s"], 3)
json.dump(res, open(f"{R}/results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
PY
log "done; results in $R/results.json"
