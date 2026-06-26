# OLMo-2-7B inference speed on the C++ engine (H100)

Single-stream decode, batch 1, measured on kuiper H100 **while another job held
~94% of the GPU** — idle numbers would be higher.

| Config | tok/s | note |
|---|---|---|
| fp32, eager | **7.2** | overhead-bound (eager kernel launches), ~16× the memory roofline |
| fp32, eager, `--bf16` | 4.4 | SLOWER — bf16 adds conversion overhead, doesn't fix the bottleneck |
| **fp32, `--cuda-graph --paged-kv`** | **47.9** | **6.6× — the win.** Graphs eliminate per-kernel launch overhead |
| `--cuda-graph --paged-kv --bf16` | (crashes) | bf16 + graph capture path incompatible — use fp32 |

## The lever is CUDA graphs, not quantization
Decode here is **overhead-bound, not memory-bound**, so weight quantization
(bf16/INT4) does little while eager; CUDA graphs are the real fix. After graphs,
decode is ~2.5× the fp32 memory roofline, so bf16/INT4 *could* add a bit more —
but the bf16+graph path currently crashes, and INT4-weight inference is not wired
(see below). Graphs alone already deliver the 6.6×.

## Reproduce
```bash
V=/media/volume/Prep_and_Voice_Training
./build/chat --checkpoint $V/olmo2-7b/olmo2_7b.pt \
  --config configs/olmo2_1124_7B.json \
  --vocab-file $V/olmo2-tok/vocab.json --merges-file $V/olmo2-tok/merges.txt \
  --device cuda --cuda-graph --paged-kv
```

## INT4 weight inference: designed, not wired
`quantize_int4` writes a **sidecar** archive (`<name>.int4.weight/.scales/.group_size`,
6.3 GB) — NOT a standard module checkpoint, so `torch::load(model, int4.pt)` in
chat/bench_chat crashes (that's the "INT4 crash"). The consuming path (load sidecar
+ int4 GEMV via `src/nn/quant.cpp` / `kernels/quant_dequant.cu`) was never wired
into chat. Given decode is overhead-bound and graphs already give 6.6×, INT4 is
**low priority** for single-stream; it matters more for batched/throughput.

## bench_chat + graphs
`bench_chat` has no paged-KV decode, and chat's graph capture requires `--paged-kv`,
so graph-bench lives in `chat` for now (numbers above). Porting paged-KV + capture
into bench_chat is a follow-up; not needed to get the headline number.
