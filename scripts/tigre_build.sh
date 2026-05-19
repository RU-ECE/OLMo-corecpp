#!/usr/bin/env bash
# Build OLMo-corecpp on tigre (RTX 5060 Ti, Blackwell sm_120).
#
# Requires CUDA 12.8 at /usr/local/cuda-12.8 (not the /usr/local/cuda symlink).
# LibTorch: pip cu128 or /opt/libtorch.
#
# Usage:
#   ./scripts/tigre_build.sh
#   CMAKE_CUDA_ARCHITECTURES="120" ./scripts/tigre_build.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build"
TIGRE_CUDA_DEFAULT="/usr/local/cuda-12.8"

nvcc_version() {
  "$1/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'
}

nvcc_ge_12_8() {
  local ver maj min
  ver="$(nvcc_version "$1")"
  maj="${ver%%.*}"
  min="${ver#*.}"
  min="${min%%.*}"
  [ "$maj" -gt 12 ] || { [ "$maj" -eq 12 ] && [ "$min" -ge 8 ]; }
}

find_cuda() {
  local c
  if [ -n "${CUDA_HOME:-}" ]; then
    if [ -x "${CUDA_HOME}/bin/nvcc" ] && nvcc_ge_12_8 "$CUDA_HOME"; then
      echo "$CUDA_HOME"
      return
    fi
    echo "ERROR: CUDA_HOME=${CUDA_HOME} must have nvcc 12.8+." >&2
    exit 1
  fi
  for c in "$TIGRE_CUDA_DEFAULT"; do
    if [ -x "${c}/bin/nvcc" ] && nvcc_ge_12_8 "$c"; then
      echo "$c"
      return
    fi
  done
  echo ""
}

find_libtorch() {
  if [ -n "${LIBTORCH_DIR:-}" ] && [ -d "$LIBTORCH_DIR" ]; then
    echo "$LIBTORCH_DIR"; return
  fi
  if [ -d /opt/libtorch ]; then
    echo /opt/libtorch; return
  fi
  local prefix
  prefix="$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || true)"
  if [ -n "$prefix" ] && [ -d "$prefix" ]; then
    echo "$prefix"; return
  fi
  echo ""
}

CUDA_HOME="$(find_cuda)"
if [ -z "$CUDA_HOME" ]; then
  echo "ERROR: install CUDA 12.8: sudo apt install -y cuda-toolkit-12-8" >&2
  echo "  Expected: ${TIGRE_CUDA_DEFAULT}/bin/nvcc" >&2
  exit 1
fi

# Drop other CUDA toolkit bins from PATH so CMake/PyTorch see one nvcc.
clean_path="$(echo "$PATH" | tr ':' '\n' | grep -v -E '/usr/local/cuda(-[0-9.]+)?/bin$' | paste -sd: -)"
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export CUDA_NVCC_EXECUTABLE="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${clean_path}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

NVCC_VER="$(nvcc_version "$CUDA_HOME")"

LIBTORCH_DIR="$(find_libtorch)"
if [ -z "$LIBTORCH_DIR" ]; then
  echo "ERROR: LibTorch not found." >&2
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cu128" >&2
  exit 1
fi

ARCHES="${CMAKE_CUDA_ARCHITECTURES:-120}"

# Stale configure cached 12.1 while /usr/local/cuda still pointed at old toolkit.
if [ -f "$BUILD_DIR/CMakeCache.txt" ] && grep -q 'CUDAToolkit_VERSION:STRING=12\.1' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
  echo "Removing stale CMake cache (CUDA 12.1)..." >&2
  rm -f "$BUILD_DIR/CMakeCache.txt"
fi

echo "=== tigre build (5060 Ti) ==="
echo "CUDA_HOME : ${CUDA_HOME} (nvcc ${NVCC_VER})"
echo "LibTorch  : ${LIBTORCH_DIR}"
echo "Arches    : ${ARCHES}"
echo

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCHES}" \
  "$@"

NPROC="$(nproc 2>/dev/null || echo 4)"
cmake --build "$BUILD_DIR" -j "$NPROC"

echo
echo "=== Build complete: ${BUILD_DIR} ==="
