#!/usr/bin/env bash
# scripts/race/01_build_cpp.sh
#
# Builds the C++ side targeting the detected GPU arch (sm_120 on the
# 5060 Ti). Fast-fails BEFORE the long compile if the toolchain can't
# target this GPU — the alternative is a 5-minute build that dies on
# the first .cu file with "unsupported gpu architecture 'sm_120'".

set -euo pipefail
cd "$(dirname "$0")/../.."

say()  { printf "\033[1;36m[build]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m  ✓\033[0m  %s\n" "$*"; }
fail() { printf "\033[1;31m  ✗\033[0m  %s\n" "$*"; exit 1; }
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# ── Resolve target GPU arch ──
# Prefer the value 00_env_check.sh persisted; else detect now; else default sm_120.
if [[ -f scripts/race/results/.gpu_arch ]]; then
  ARCH=$(cat scripts/race/results/.gpu_arch)
elif command -v nvidia-smi >/dev/null; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
else
  ARCH=120
fi
say "target GPU arch: sm_${ARCH}"

# ── Pre-build toolchain guard (the Blackwell floor) ──
if   [[ "$ARCH" -ge 100 ]]; then CUDA_FLOOR="12.8"
elif [[ "$ARCH" -ge 90  ]]; then CUDA_FLOOR="12.0"
else                              CUDA_FLOOR="11.8"
fi

command -v nvcc >/dev/null || fail "nvcc not on PATH — need CUDA Toolkit ≥ ${CUDA_FLOOR} for sm_${ARCH}"
NVCC_VER=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | head -1 | awk '{print $2}')
if ver_ge "$NVCC_VER" "$CUDA_FLOOR"; then
  ok "nvcc $NVCC_VER (≥ ${CUDA_FLOOR})"
else
  fail "nvcc $NVCC_VER too old for sm_${ARCH}. Need ≥ ${CUDA_FLOOR}.
       (sm_120 / Blackwell needs CUDA 12.8.)  Run scripts/race/00_env_check.sh
       for the full diagnosis."
fi

# torch must exist AND ship kernels for this arch.
python3 -c "import torch" 2>/dev/null || fail "torch not installed (pip3 install torch --index-url https://download.pytorch.org/whl/cu128)"
TORCH_OK=$(python3 - "$ARCH" <<'PY'
import sys, torch
arch = sys.argv[1]
al = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
print("yes" if (f"sm_{arch}" in al or f"compute_{arch}" in al) else "no")
PY
)
[[ "$TORCH_OK" == "yes" ]] || fail "installed torch has no sm_${ARCH} kernels.
       For Blackwell:  pip3 install --upgrade torch --index-url https://download.pytorch.org/whl/cu128"
ok "torch ships sm_${ARCH} kernels"

# ── Locate LibTorch ──
LIBTORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
[[ -d "$LIBTORCH_PATH" ]] || fail "libtorch cmake dir not found via pip torch"
ok "libtorch: $LIBTORCH_PATH"

# ── Configure + build ──
say "configuring (Release, sm_${ARCH}, kernels ON)"
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DOLMO_BUILD_KERNELS=ON \
  2>&1 | tail -25

say "compiling with $(nproc) jobs (first build ~5 min)"
make -j"$(nproc)" 2>&1 | tail -20

say "build complete"
for t in olmo_train chat prepare_data test_cuda_parity test_fused_ce test_fused_qkv_rope; do
  if [[ -x "$t" ]]; then ok "$t"; else printf "\033[1;33m  ⚠ missing: %s\033[0m\n" "$t"; fi
done
