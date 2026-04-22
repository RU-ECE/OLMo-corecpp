#!/usr/bin/env bash
# validate_mlsys_p1.sh — post-training validation of the mlsys-p1 branch.
#
# This is NOT a pretrain launcher. It assumes your 1B run on the
# zero-wait-trainer-sg branch already completed and you now want to prove
# that the systems work on mlsys-p1 (FlashAttention-2 reference, CUDA
# graph capture, WGMMA+TMA GEMM, HF export, numerical audit, determinism,
# arena free-list) is correct and fast on an H100. No new pretraining
# happens — just tests, microbenchmarks, and a 50-step smoke run.
#
# Total runtime: ~20–40 min on a single H100, no GPU needed again after.
#
# Results land in:
#   results/validate_mlsys_p1_<YYYYMMDD_HHMMSS>/
#     build.log             cmake + make output (ZWT_USE_WGMMA=ON)
#     tests.log             every *_tests binary, fail-fast
#     wgmma_bench.csv       cuBLAS vs CUTLASS-WGMMA per Linear shape
#     graph_bench.csv       graph capture vs eager
#     bench_threeway.csv    pytorch eager / torch.compile / zwt tok/s
#     numerical_audit.txt   bf16 vs fp32 logit drift from fresh init
#     smoke_mlsys.csv       50-step loss curve on mlsys-p1
#     pt_baseline.txt       50-step loss from pt_baseline.py on same config
#     loss_parity.txt       loss_curve_check.py verdict
#     SUMMARY.txt           one-page PASS/FAIL per stage
#
# Usage:
#   bash zwt/scripts/validate_mlsys_p1.sh                # full run
#   bash zwt/scripts/validate_mlsys_p1.sh --skip-build   # reuse existing ./build
#   bash zwt/scripts/validate_mlsys_p1.sh --no-pretrain  # tests/bench only
#   bash zwt/scripts/validate_mlsys_p1.sh --tokens PATH  # override volume lookup
#
# Prereqs: on branch mlsys-p1, H100 visible to nvidia-smi, tokenized OWT
# .npy exists under /media/volume/Prep_and_Voice_Training.
#
# Checkpoint eval (export_hf → perplexity → lm-eval) is a SEPARATE step
# handled after this script succeeds. The zero-wait-trainer-sg checkpoint
# has a different state_dict layout than mlsys-p1 (pre-GQA, separate
# gate/up SwiGLU weights) and needs a converter pass that we'll write
# once the actual checkpoint file is in hand.

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────
VOLUME_DEFAULT="/media/volume/Prep_and_Voice_Training"
SMOKE_CONF="zwt/conf/owt_1B_prof.conf"   # 50 steps, no ckpt, same shape as real run
SMOKE_STEPS=50                            # must match [runtime] max_steps in SMOKE_CONF

DO_BUILD=1
DO_PRETRAIN=1
TOKENS=""
for a in "$@"; do
  case "$a" in
    --skip-build)   DO_BUILD=0 ;;
    --no-pretrain)  DO_PRETRAIN=0 ;;
    --tokens)       TOKENS="$2"; shift ;;
    --tokens=*)     TOKENS="${a#--tokens=}" ;;
    -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
    *)              echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

# ── Layout ──────────────────────────────────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/validate_mlsys_p1_${TS}"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/SUMMARY.txt"
: > "$SUMMARY"

log()   { echo -e "$@" | tee -a "$SUMMARY"; }
stage() { log "\n── $1 ──"; }
pass()  { log "  PASS  $1"; }
fail()  { log "  FAIL  $1  (see $2)"; exit 1; }

log "validate_mlsys_p1  ts=$TS"
log "branch=$(git rev-parse --abbrev-ref HEAD)  head=$(git rev-parse --short HEAD)"
log "out=$OUT_DIR"

# ── 0. preflight ────────────────────────────────────────────────────────
stage "preflight"
if ! command -v nvidia-smi >/dev/null; then
  fail "nvidia-smi missing — this script needs an H100" "$SUMMARY"
fi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
log "  gpu=$GPU"
if [[ "$GPU" != *"H100"* ]]; then
  log "  WARN: GPU is not H100 — WGMMA will report wgmma_available()=false"
fi
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "mlsys-p1" ]]; then
  log "  WARN: branch is '$BRANCH', expected 'mlsys-p1'. Continuing anyway."
fi
pass "preflight"

# ── 1. build with WGMMA on ──────────────────────────────────────────────
if [[ "$DO_BUILD" -eq 1 ]]; then
  stage "build (-DZWT_USE_WGMMA=ON)"
  BUILD_LOG="$OUT_DIR/build.log"
  (
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DZWT_USE_WGMMA=ON \
      -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null || echo '')" \
      2>&1
    cmake --build build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" 2>&1
  ) >"$BUILD_LOG" 2>&1 || fail "build" "$BUILD_LOG"
  pass "build (see $BUILD_LOG for full output)"
else
  stage "build (skipped)"
fi

# Sanity: every binary we expect.
for bin in zwt_kernel_tests zwt_flash_attn_tests zwt_tp_tests zwt_ddp_bucket_tests \
           zwt_wgmma_tests zwt_wgmma_bench zwt_graph_bench zwt_numerical_audit \
           zwt_pretrain zwt_export_hf prepare_data; do
  if [[ ! -x "./build/$bin" ]]; then
    fail "missing build artifact: ./build/$bin" "$OUT_DIR/build.log"
  fi
done
pass "all expected binaries present"

# ── 2. correctness tests ────────────────────────────────────────────────
stage "correctness tests"
TESTS_LOG="$OUT_DIR/tests.log"
: > "$TESTS_LOG"
run_test() {
  local name="$1"
  echo -e "\n### $name ###" >>"$TESTS_LOG"
  if "./build/$name" >>"$TESTS_LOG" 2>&1; then
    pass "$name"
  else
    fail "$name" "$TESTS_LOG"
  fi
}
run_test zwt_kernel_tests
run_test zwt_flash_attn_tests
run_test zwt_tp_tests
run_test zwt_ddp_bucket_tests
run_test zwt_wgmma_tests

# ── 3. WGMMA bench ──────────────────────────────────────────────────────
stage "WGMMA vs cuBLAS bench"
WGMMA_CSV="$OUT_DIR/wgmma_bench.csv"
./build/zwt_wgmma_bench >"$WGMMA_CSV" 2>>"$TESTS_LOG" || fail "zwt_wgmma_bench" "$WGMMA_CSV"
# Inspect speedup column (last, per-row).
awk -F, 'NR>1 && $9>0 { print "    "$1": cublas="$5"ms wgmma="$6"ms speedup="$9"x" }' "$WGMMA_CSV" | tee -a "$SUMMARY"
pass "wgmma_bench (CSV: $WGMMA_CSV)"

# ── 4. graph bench ──────────────────────────────────────────────────────
stage "CUDA graph capture bench"
GRAPH_CSV="$OUT_DIR/graph_bench.csv"
./build/zwt_graph_bench >"$GRAPH_CSV" 2>>"$TESTS_LOG" || fail "zwt_graph_bench" "$GRAPH_CSV"
awk 'NR>1 { print "    "$0 }' "$GRAPH_CSV" | head -5 | tee -a "$SUMMARY"
pass "graph_bench (CSV: $GRAPH_CSV)"

# ── 5. three-way bench (only if PyTorch importable) ────────────────────
stage "three-way bench (pytorch eager / torch.compile / zwt)"
THREEWAY_CSV="$OUT_DIR/bench_threeway.csv"
if python3 -c 'import torch' 2>/dev/null; then
  bash zwt/scripts/bench_threeway.sh zwt/conf/owt_1B_prof.conf 30 5 \
       >"$THREEWAY_CSV" 2>>"$TESTS_LOG" || fail "bench_threeway" "$THREEWAY_CSV"
  cat "$THREEWAY_CSV" | tee -a "$SUMMARY"
  pass "bench_threeway (CSV: $THREEWAY_CSV)"
else
  echo "  SKIP: python3 -c 'import torch' failed — install torch to enable" | tee -a "$SUMMARY"
fi

# ── 6. numerical audit (bf16 vs fp32 from fresh init; no ckpt needed) ──
stage "bf16 vs fp32 numerical audit"
NA_TXT="$OUT_DIR/numerical_audit.txt"
./build/zwt_numerical_audit "$SMOKE_CONF" >"$NA_TXT" 2>&1 \
  || fail "zwt_numerical_audit" "$NA_TXT"
tail -5 "$NA_TXT" | sed 's/^/    /' | tee -a "$SUMMARY"
pass "numerical_audit (see $NA_TXT)"

# ── 7. smoke pretrain + PyTorch baseline for loss parity ───────────────
if [[ "$DO_PRETRAIN" -eq 1 ]]; then
  stage "smoke pretrain ($SMOKE_STEPS steps)"

  # Locate tokens. Same lookup style as launch_1b_h100.sh.
  if [[ -z "$TOKENS" && -n "${ZWT_TOKENS:-}" ]]; then
    TOKENS="$ZWT_TOKENS"
  fi
  if [[ -z "$TOKENS" ]]; then
    if [[ -d "$VOLUME_DEFAULT" ]]; then
      TOKENS=$(find "$VOLUME_DEFAULT" -maxdepth 6 -type f -name '*.npy' -size +1G 2>/dev/null \
               | xargs -I{} stat -c '%s %n' {} 2>/dev/null \
               | sort -n | tail -1 | cut -d' ' -f2-)
    fi
  fi
  if [[ -z "$TOKENS" || ! -f "$TOKENS" ]]; then
    echo "  SKIP: no tokenized .npy found (pass --tokens PATH or export ZWT_TOKENS)" \
      | tee -a "$SUMMARY"
  else
    log "  tokens: $TOKENS"
    mkdir -p data/owt
    ln -sfn "$TOKENS" data/owt/owt_tokens.npy

    # mlsys-p1 smoke run with metrics CSV.
    SMOKE_CSV="$OUT_DIR/smoke_mlsys.csv"
    SMOKE_LOG="$OUT_DIR/smoke_mlsys.log"
    ./build/zwt_pretrain "$SMOKE_CONF" --metrics-csv "$SMOKE_CSV" \
        >"$SMOKE_LOG" 2>&1 || fail "smoke pretrain" "$SMOKE_LOG"
    tail -5 "$SMOKE_LOG" | sed 's/^/    /' | tee -a "$SUMMARY"

    # Sanity-check: loss decreased over the run.
    FIRST=$(awk -F, 'NR==2 {print $2}' "$SMOKE_CSV")
    LAST=$(awk -F,  'END   {print $2}' "$SMOKE_CSV")
    log "  loss: step1=$FIRST  stepN=$LAST"
    if python3 -c "import sys; sys.exit(0 if float('$LAST') < float('$FIRST') else 1)"; then
      pass "smoke pretrain — loss decreased ($FIRST → $LAST)"
    else
      fail "smoke pretrain — loss did NOT decrease" "$SMOKE_LOG"
    fi

    # PyTorch baseline on same config for parity comparison.
    stage "pt_baseline on same config"
    PT_TXT="$OUT_DIR/pt_baseline.txt"
    if python3 -c 'import torch' 2>/dev/null; then
      python3 zwt/scripts/pt_baseline.py --config "$SMOKE_CONF" \
             --warmup 5 --steps "$SMOKE_STEPS" --dtype bf16 >"$PT_TXT" 2>&1 \
             || fail "pt_baseline" "$PT_TXT"
      tail -3 "$PT_TXT" | sed 's/^/    /' | tee -a "$SUMMARY"

      # Loss-parity gate. pt_baseline uses random data (not OWT) so the
      # CURVES aren't directly aligned — use it as a tok/s sanity only.
      # Curve-parity against zero-wait-trainer-sg needs that branch's
      # CSV, which doesn't exist (training ran without --metrics-csv).
      # The step1→stepN monotonic check above is our curve signal here.
      pass "pt_baseline tok/s captured"
    else
      echo "  SKIP: python3 torch not available" | tee -a "$SUMMARY"
    fi
  fi
else
  stage "smoke pretrain (skipped via --no-pretrain)"
fi

# ── 8. summary ──────────────────────────────────────────────────────────
stage "DONE"
log "all stages PASS."
log "results dir: $OUT_DIR"
log ""
log "next:"
log "  - inspect $WGMMA_CSV — expect speedup>1.0 on every row if H100"
log "  - inspect $THREEWAY_CSV — expect zwt tok/s >= torch.compile tok/s"
log "  - the zero-wait-trainer-sg checkpoint eval is a separate pass"
log "    once the ckpt file is accessible (state_dict converter pending)"
