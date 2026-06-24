# Professor Runbook — Fastest Inference + Benchmark vs ollama & llama.cpp

The trained **1B model** (973M params, d=2048, 16 layers) and everything needed
to run + benchmark it. Verified working: **55 tok/s single-stream on an M4 Pro
(MPS, fp32)** — your CUDA box (RTX 5060 Ti) will be faster in bf16/INT4.

Files (in this repo):
- `runs/kuiper_1B/model.pt` — the trained model (1.95 GB; see step 0 to fetch).
- `configs/kuiper_1B.json` — architecture config (must match the model exactly).
- `data/gpt2/{vocab.json,merges.txt}` — GPT-2 tokenizer.

---

## 0. Get the model onto the box
It lives on the kuiper volume. From a machine with SSH access:
```bash
mkdir -p runs/kuiper_1B
scp prepxl:/media/volume/Prep_and_Voice_Training/runs/1B/model.pt runs/kuiper_1B/model.pt
```

## 1. Build
```bash
./scripts/build.sh --cuda           # CUDA box (RTX 5060 Ti = sm_120, needs CUDA 12.8+)
# (Mac: ./scripts/build.sh   ·  CPU-only: ./scripts/build.sh --cpu)
```

## 2. Fastest inference (interactive chat)
```bash
./build/chat \
  --checkpoint runs/kuiper_1B/model.pt \
  --config configs/kuiper_1B.json \
  --vocab-file data/gpt2/vocab.json \
  --merges-file data/gpt2/merges.txt \
  --device cuda
```
It's a **base model** (not chat-tuned, and only ~2B training tokens) — prompt it
**completion-style**: type `The capital of France is` and let it continue.
On CUDA the model loads in **bf16** automatically (fastest); the loader handles
the bf16 checkpoint (no conversion needed).

**Even faster — INT4:** quantize once, then point chat/bench at the INT4 file:
```bash
./build/quantize_int4 --in runs/kuiper_1B/model.pt \
  --out runs/kuiper_1B/model.int4.pt --config configs/kuiper_1B.json
# then use --checkpoint runs/kuiper_1B/model.int4.pt
```

## 3. Benchmark — our C++ engine vs ollama vs llama.cpp
One command (skips engines that aren't installed):
```bash
CKPT=runs/kuiper_1B/model.pt CONFIG=configs/kuiper_1B.json DEVICE=cuda \
  bash scripts/bench/bench_1b_infer.sh
```
It reports:
- **Our C++ engine** on YOUR 1B at batch 1 / 8 / 32 (bf16 + INT4).
- **ollama** (`llama3.2:1b`, Q4) single-stream eval rate.
- **llama.cpp** single-stream (set `LLAMACPP_GGUF=/path/to/a/1B.gguf`).

**How to read it (important for the paper):**
- **batch-1 row** = single-stream latency — compare our **INT4** vs ollama/llama.cpp
  **Q4** (both 4-bit; that's the apples-to-apples single-stream number).
- **batch-8/32 rows** = THROUGHPUT — this is our edge: a batched server saturates
  the GPU; ollama/llama.cpp single-stream can't match aggregate tok/s.

Why ollama/llama.cpp run a *standard* 1B (llama3.2:1b) and not our exact weights:
they need **GGUF**, and our custom architecture (MTP heads, reordered norm) doesn't
convert cleanly. Matching at **1B scale** (same FLOPs/token) is the fair
engine-speed comparison. To force the *exact same weights*, see §4.

## 4. (Optional, advanced) Same-weights GGUF for ollama/llama.cpp
Only if you want ollama/llama.cpp running the *identical* trained weights:
```bash
# a) export our .pt -> HF safetensors (architecture must map to OLMo-2):
./build/zwt_export_hf --checkpoint runs/kuiper_1B/model.pt \
  --config configs/kuiper_1B.json --out runs/kuiper_1B/hf/
# b) HF -> GGUF with llama.cpp's converter:
python llama.cpp/convert_hf_to_gguf.py runs/kuiper_1B/hf/ --outfile kuiper_1b.gguf
# c) quantize + run:
llama.cpp/llama-quantize kuiper_1b.gguf kuiper_1b.Q4_K_M.gguf Q4_K_M
LLAMACPP_GGUF=kuiper_1b.Q4_K_M.gguf bash scripts/bench/bench_1b_infer.sh
```
**Caveat (untested):** the MTP heads + reordered-norm block may not round-trip
through llama.cpp's OLMo-2 converter. If the export/convert errors, fall back to
§3 (matched-scale) — the speed conclusion is the same.

---

### Reference numbers
| Engine | Model | Device | Single-stream |
|---|---|---|---|
| **Our C++** | trained 1B, fp32 | M4 Pro (MPS) | **55.3 tok/s** (verified) |
| Our C++ | trained 1B, bf16/INT4 | RTX 5060 Ti | run step 3 |
| ollama | llama3.2:1b Q4 | RTX 5060 Ti | run step 3 |
| llama.cpp | 1B Q4 | RTX 5060 Ti | run step 3 |

Report **steady-state** tok/s (not the cumulative average the trainer prints).
