#!/usr/bin/env bash
# GPU proof tests for the speed plan, written for a Google Colab GPU runtime.
#
#   unzip llm-cpp.zip -d /content && bash /content/llm-cpp/scripts/colab_gpu_tests.sh
#
# Builds only the test targets against the pip torch already in Colab, for the
# GPU that is attached, then runs:
#   test_grad_coverage cuda    bug 1: every trainable parameter gets a gradient
#   (bug 1 planted back)       the same test must FAIL with the old fast_linear
#   test_spec_greedy           bug 5: speculative == greedy (CPU, sanity)
#   test_slmtp                 SL-MTP factored == direct (CPU)
#   test_ddp_accum             bug 2: two-rank DDP with grad accumulation (CPU)
#   zwt_ddp_loopback_tests     bug 3: zwt records every bucket every step
#   bench_slmtp --device cuda  SL-MTP vs full heads, 1B and 7B shapes, fp32 + bf16
# Everything is logged to /content/gpu_results.txt.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=/content/gpu_results.txt
: > "$OUT"
log() { echo "$@" | tee -a "$OUT"; }
run() { log "\$ $*"; "$@" 2>&1 | tee -a "$OUT"; log "exit ${PIPESTATUS[0]}"; log ""; }

cd "$ROOT"
run nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
run nvcc --version
pip -q install ninja >/dev/null 2>&1 || true

ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
TARGETS="test_grad_coverage test_spec_greedy test_slmtp test_ddp_accum zwt_ddp_loopback_tests bench_slmtp test_dora_factored"
log "== configure (sm_$ARCH)"
cmake -G Ninja -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DOLMO_RACE_BUILD=OFF >> "$OUT" 2>&1 \
  || { log "configure FAILED"; tail -40 "$OUT"; exit 1; }
log "== build ($TARGETS)"
start=$(date +%s)
ninja -C build-cuda -j"$(nproc)" $TARGETS >> "$OUT" 2>&1 || { log "build FAILED"; grep -n 'error' "$OUT" | head -30; exit 1; }
log "build took $(( $(date +%s) - start )) s"

B=build-cuda
run $B/test_grad_coverage cuda cpu
log "== bug 1 planted back: fast_linear without the autograd check (must FAIL)"
cp src/backend/cublas_direct.cpp /tmp/cublas_direct.cpp.fixed
sed -i 's/      \&\& !records_grad({x, weight, bias})) {/      ) {  \/\/ PLANTED: bug 1/' src/backend/cublas_direct.cpp
grep -c 'PLANTED' src/backend/cublas_direct.cpp | sed 's/^/planted sites: /' | tee -a "$OUT"
ninja -C $B -j"$(nproc)" test_grad_coverage >> "$OUT" 2>&1
run $B/test_grad_coverage cuda
cp /tmp/cublas_direct.cpp.fixed src/backend/cublas_direct.cpp
ninja -C $B -j"$(nproc)" test_grad_coverage >> "$OUT" 2>&1
run $B/test_grad_coverage cuda

run $B/test_spec_greedy
run $B/test_slmtp
run $B/test_ddp_accum
run $B/zwt_ddp_loopback_tests
run $B/test_dora_factored --device cuda
run $B/test_dora_factored --device cuda --tokens 4096 --out 4096 --in 4096
for dt in "" "--bf16"; do
  run $B/bench_slmtp --device cuda --d 2048 --vocab 50304 --tokens 2048 --heads 3 --rank 64 --iters 10 $dt
  run $B/bench_slmtp --device cuda --d 4096 --vocab 100352 --tokens 1024 --heads 3 --rank 64 --iters 10 $dt
done
log "== SL-MTP chunked loss without the CUDA row kernels (reference path), bf16"
export OLMO_SLMTP_NO_KERNEL=1
run $B/bench_slmtp --device cuda --d 2048 --vocab 50304 --tokens 2048 --heads 3 --rank 64 --iters 10 --bf16
run $B/bench_slmtp --device cuda --d 4096 --vocab 100352 --tokens 1024 --heads 3 --rank 64 --iters 10 --bf16
unset OLMO_SLMTP_NO_KERNEL
log "== done"
