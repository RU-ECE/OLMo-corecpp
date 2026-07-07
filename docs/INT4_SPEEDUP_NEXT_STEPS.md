# INT4 Inference Speedup — Next Steps & Runbook

Goal: beat ollama/llama.cpp on **single-stream (batch-1) int4 decode** of the 1B
model, on the RTX 5060 Ti (Blackwell, sm_120, ~448 GB/s).

## Where we are

| Config (batch 1, single stream) | tok/s | notes |
|---|---|---|
| ollama `llama3.2:1b` Q4 | **258** | the target to beat (fair: both 4-bit, 1 stream) |
| our int4 — old kernel + fp32 LM head | 89.6 | starting point |
| our int4 — old kernel + **int4 LM head** | 93.5 | LM-head quant shipped (small win: LM head was already cuBLAS-efficient) |
| our int4 — **fast kernel v3** | **← run the sweep below** | hoisted scale + register extract + tunable occupancy |
| our fp32 (cuBLAS) | 65.6 | full-precision baseline |

Roofline reality on the 5060 Ti (~448 GB/s): ollama is at ~42% of roofline, our
old int4 kernel is at ~9%. **The entire gap is kernel efficiency** — there is
~11× of unused headroom, so the kernel is where the win is.

> ⚠️ Do **not** claim "we beat ollama" from the batch-32 throughput number
> (391 tok/s). That is 32 concurrent streams vs ollama's 1 — not a fair
> comparison (llama.cpp batches too). The fair, defensible number is **batch-1
> int4 vs ollama Q4**.

## What has shipped (branch `2fast2furious`)

1. **`bench_chat` uses CUDA graphs + paged-KV** at batch-1 on CUDA (the 6.6× path,
   previously only in `chat`). Default on; `--eager` forces the baseline.
2. **fp32, not bf16, for the graph path** — bf16 + CUDA-graph is unsupported
   (the graph-safe paged-KV write kernel needs fp32 pools).
3. **LM head quantized to int4** — was a 412 MB/token fp32 read; reuses the
   int4 path. `enable_int4` is backward-compatible.
4. **`quantize_int4` runs on any machine** — loads on CPU + bf16-aware.
5. **Fast int4 GEMV kernel `int4_awq_gemv_fast_kernel`** (opt-in
   `OLMO_INT4_FAST=1`): warp-per-row, vectorized `uint4` loads, register nibble
   extraction, **hoisted per-group scale**, **runtime-tunable warps/block**
   (`OLMO_INT4_RPB`). Falls back to the reference kernel when `group_size%32!=0`.

---

## STEP 1 — Rebuild and re-quantize (once)

```bash
cd ~/git/research/OLMo-corecpp
git pull
./scripts/build.sh --cuda
rm -f runs/kuiper_1B/model.int4.pt      # force re-quantize WITH the LM head
```

## STEP 2 — Sweep the fast kernel's occupancy (the money run)

```bash
for rpb in 2 4 8 16 32; do
  printf "RPB=%s  " "$rpb"
  OLMO_INT4_FAST=1 OLMO_INT4_RPB=$rpb ./build/bench_chat \
    --int4 runs/kuiper_1B/model.int4.pt --config configs/kuiper_1B.json \
    --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt \
    --device cuda --batch 1 --prompt-len 128 --decode-len 256 --warmup 1 --iters 3 \
    2>&1 | grep Throughput
done
```

- Record the **best** `RPB` and its tok/s. That is the int4 single-stream number.
- The `p50` in the TPOT line is the steady-state; `1000/p50` is the cleanest rate.
- Sanity-check correctness once (should be coherent English, not garbage):
  ```bash
  printf 'Once upon a time' | OLMO_INT4_FAST=1 ./build/chat \
    --int4 runs/kuiper_1B/model.int4.pt --config configs/kuiper_1B.json \
    --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt \
    --device cuda --max-tokens 30 --temperature 0
  ```

**Decision:**
- **Best RPB ≥ 258 tok/s** → we beat ollama single-stream. Make that `RPB` the
  default in `int4_gemv_cuda` and lock it in. Done.
- **Best RPB < 258** → go to Step 3 (profile), don't keep guessing.

## STEP 3 — Profile the kernel (only if Step 2 < 258)

```bash
ncu --set full -k int4_awq_gemv_fast_kernel -c 1 -o int4prof \
  bash -c 'OLMO_INT4_FAST=1 ./build/bench_chat --int4 runs/kuiper_1B/model.int4.pt \
  --config configs/kuiper_1B.json --vocab-file data/gpt2/vocab.json \
  --merges-file data/gpt2/merges.txt --device cuda --batch 1 --decode-len 32'
```

Send back (or read from `ncu-ui int4prof.ncu-rep`):
- **Memory Throughput %** of peak (are we bandwidth-bound yet?)
- **Achieved Occupancy**
- **Warp State / top stall reason** (LG throttle, long scoreboard, barrier, etc.)

These three numbers say exactly which lever is next, instead of guessing.

## STEP 4 — Remaining optimization levers (in likely-ROI order)

1. **`dp4a` int8 dot-product** — llama.cpp's core trick. Quantize the activation
   `x` to int8 per group on the fly, then `__dp4a` does 4 INT8 MACs per
   instruction and reads `x` at 1/4 the bytes. Biggest remaining lever if the
   profile shows we're compute/instruction-bound rather than bandwidth-bound.
2. **Quantize the embedding** (`tokens_embed`) — currently fp32 (412 MB). Only a
   VRAM win for decode (it's a row lookup, not a GEMM), but frees memory on the
   8 GB card and lets bigger batches fit.
3. **Fuse RMSNorm + int4 GEMV** for QKV / gate-up to cut a kernel launch + a
   round-trip to HBM per block.
4. **Persistent-kernel / split-K** for the LM head (50304 rows) if the profile
   shows it dominates.
5. **MTP self-speculative decode** — architectural edge ollama's llama3.2 model
   does not have. Currently 0% accept under int4 (quantization desyncs the MTP
   heads); revisit with a higher-precision draft or fp32 MTP heads.

## Known issues / gotchas

- **OOM at batch 32 on the shared box:** other processes (leftover runs, ollama's
  resident model) hold VRAM. Fixes:
  ```bash
  pkill -f bench_chat                       # kill crashed leftovers
  sudo systemctl stop ollama                # (optional) free ollama's VRAM
  # or just skip the big batch — batch-1 is the number that matters:
  BATCHES="1 8" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True <bench cmd>
  ```
- **bf16 + CUDA graph crashes** (`paged_kv_write_dyn` fp32-pool check). Use fp32
  or int4 for the graph path — the bench already does.
- **No `nvcc` on the Mac dev box** — every kernel change compiles first on the
  GPU box. If a build errors, paste the compiler output for a one-shot fix.

## Honest framing for the paper

The defensible claims, regardless of whether we clear 258:
1. **6× internal decode speedup** from CUDA graphs + paged-KV (measured: eager
   ~15 → 90+ tok/s), plus the kernel work on top.
2. **We run an architecture llama.cpp/ollama cannot** (MTP heads, reordered
   norm, DC-MRE) — a scale-matched engine comparison, not weight-identical.
3. **An honest kernel-efficiency decomposition**: where llama.cpp's remaining
   lead comes from (fused Q4 kernels, dp4a, quantized embed/LM-head) — a
   contribution in itself, and the roadmap above closes it.

Do not publish the batch-32-vs-single-stream comparison as a win.
