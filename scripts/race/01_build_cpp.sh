#!/usr/bin/env bash
# scripts/race/01_build_cpp.sh
#
# Builds the C++ side targeting sm_120 (5060 Ti). All warnings, full opt,
# kernels included. Caches the build dir so repeat runs are fast.

set -euo pipefail
cd "$(dirname "$0")/../.."

say() { printf "\033[1;36m[build]\033[0m %s\n" "$*"; }
say "building llm-cpp for sm_120 in build/"

# Auto-detect LibTorch from pip-installed torch.
LIBTORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
[[ -d "$LIBTORCH_PATH" ]] || { echo "libtorch not found via pip torch"; exit 1; }
say "libtorch at: $LIBTORCH_PATH"

mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DOLMO_BUILD_KERNELS=ON \
  2>&1 | tail -30

make -j"$(nproc)" 2>&1 | tail -20

say "build complete"
ls -la olmo_train chat prepare_data test_cuda_parity test_fused_ce test_fused_qkv_rope 2>/dev/null | awk '{print "  ", $9, $5}'
