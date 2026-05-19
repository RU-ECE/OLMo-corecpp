# 250M Race — C++ vs Python on the 5060 Ti

End-to-end pipeline that builds, validates, and benchmarks the C++
implementation against a stock PyTorch baseline at matched 250M
architecture. Both sides train the same backbone (16 layers, d=1024,
ffn=2816, vocab=50257, tied embeddings); the C++ side keeps its MTP
heads on top, which is what wins inference.

## One-command run

```bash
cd llm-cpp
bash scripts/race/run_all.sh
```

The full pipeline takes ~60–90 minutes on a 5060 Ti: ~5 min for env
checks + build, ~5 min for correctness, ~5 min to download/tokenize
TinyStories on first run, ~25 min per side for training, ~2 min for
inference races, then analysis.

After it finishes, read:

```
scripts/race/results/RESULT.md
scripts/race/results/loss_curve.png    (if matplotlib is installed)
scripts/race/results/throughput.png
```

## Phase-by-phase

Every phase logs to `scripts/race/results/<phase>/`. You can skip
phases that already succeeded:

```bash
SKIP_PHASES="00 01 03" bash scripts/race/run_all.sh
```

| phase | script | what it does |
|---|---|---|
| 00 | `00_env_check.sh` | nvidia-smi, sm_120, cmake/nvcc, pytorch+cuda, disk |
| 01 | `01_build_cpp.sh` | `cmake -DCMAKE_CUDA_ARCHITECTURES=120` + parallel build |
| 02 | `02_verify_correctness.sh` | runs `test_paged_kv`, `test_prefix_cache`, `test_scheduler`, `test_fused_ce`, `test_fused_qkv_rope`, and **`test_cuda_parity`** (the GPU validation of the 3 correctness fixes in the audit) |
| 03 | `03_prepare_data.sh` | downloads TinyStories, tokenizes with GPT-2 BPE → `data/race_tokens.npy` |
| 04 | `04_train_cpp.sh` | trains C++ 250M w/ MTP, parses step/loss/tok-s to `metrics.csv` |
| 05 | `05_train_python.sh` | trains Python 250M backbone (no MTP) via OLMo-core's training script |
| 06 | `06_infer_cpp.sh` | 5 trials × 256 tokens, MTP speculative + paged KV, reports median tok/s |
| 07 | `07_infer_python.sh` | 5 trials × 256 tokens, vanilla PyTorch generation, reports median tok/s |
| 08 | `08_analyze.py` | reads all metrics, writes `RESULT.md` + plots |

## Configs

- `configs/race_250m_cpp.conf` — C++ side. SwiGLU FFN, RMSNorm, RoPE,
  tied embeddings, `num_mtp_heads=3`, no FP8 / SubQ / structural /
  multi-res / GQA — only MTP on top of the backbone.
- `configs/race_250m_python.yaml` — Python side. Same backbone. No
  MTP. Field names match OLMo-core's TransformerConfig/TrainerConfig.

Total parameter count:
- C++:  ~257 M backbone + ~3.2 M MTP heads  ≈ **260 M**
- Python:  ~257 M backbone (matches C++ backbone exactly)

## Python training entry point

`05_train_python.sh` auto-detects three common OLMo-core invocations:

1. `OLMo-corecpp/src/scripts/train.py` (script path)
2. `OLMo-corecpp/scripts/train.py` (older path)
3. `python -m olmo_core.train` (installed package)

If your OLMo-core ships a different entry point, override:

```bash
export OLMO_TRAIN_CMD="python my_train_script.py"
bash scripts/race/run_all.sh
```

If the log format differs from what the metrics parser expects, set
`OLMO_LOG_REGEX` to a Python regex with three groups: `(step, loss, tok_per_s)`.

## Python inference baseline

`python_inference_baseline.py` is a self-contained PyTorch generation
loop matching the YAML config exactly. It loads the Python-trained
checkpoint, generates 256 tokens greedily with a KV cache, and prints
the same `[N tokens, X tok/s]` trailer the C++ chat tool prints. The
analyzer scrapes both with the same regex.

To benchmark vllm or HuggingFace `generate()` instead, set:

```bash
export PY_INFER_CMD="python my_vllm_runner.py"
bash scripts/race/07_infer_python.sh
```

The runner must accept `--checkpoint`, `--prompt`, `--max-new-tokens`,
`--device`, and emit a trailing `[N tokens, X tok/s]` line.

## What "perfect" means here

- **Determinism:** both sides pin seeds and use the same tokenized
  data. Loss curves are directly comparable.
- **Correctness gate:** training cannot proceed until `test_cuda_parity`
  passes on real hardware. The three correctness fixes shipped in the
  recent audit (WMMA UB, half-rotation inverse RoPE, missing
  AutogradCUDA wrappers) are validated against ATen references first.
- **Repeatability:** every phase writes its own log + CSV under
  `scripts/race/results/<phase>/`. Re-running is `bash run_all.sh`;
  skip already-passed phases with `SKIP_PHASES`.
- **Fair comparison:** matched backbone, matched batch, matched lr,
  matched data. MTP is the only architectural asymmetry — that's what
  the inference race highlights.

## Common failures

- **`test_cuda_parity` fails on `forward y` for fused FFN:** the WMMA
  UB fix or the AutogradCUDA wrapper isn't picked up. Verify your build
  used `OLMO_BUILD_KERNELS=ON` and you're on commit `ef689f4` or later.
- **`05_train_python.sh` exits "OLMo-core training entry not found":**
  set `OLMO_TRAIN_CMD` to whatever your OLMo-core install uses.
- **Python inference baseline can't load checkpoint:** the format may
  not match. The baseline expects either `{"model": state_dict}` or a
  bare state_dict, with keys matching its `Transformer` class.
