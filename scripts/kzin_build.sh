#!/usr/bin/env bash
# Build OLMo-corecpp on the kzin/tigre box.
#
# - Auto-detects a CUDA toolkit that actually contains bin/nvcc.
#   Search order: $CUDA_HOME (if valid) -> /usr/local/cuda-12.8 -> 12.6 -> 12.4
#   -> 12.1 -> /usr/local/cuda symlink -> `which nvcc`.
# - Pins CUDA arch list. Blackwell (RTX 50-series, including 5060 Ti) is sm_120
#   and is NOT in the project's default list (80;86;89;90). Override here.
# - Configures AND builds (the previous version only configured).
#
# Overrides via env:
#   CUDA_HOME=/usr/local/cuda-12.8 ./scripts/kzin_build.sh
#   LIBTORCH_DIR=/opt/libtorch     ./scripts/kzin_build.sh
#   CMAKE_CUDA_ARCHITECTURES="120" ./scripts/kzin_build.sh   # 5060 Ti only

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build"

# ── 1. Find a CUDA toolkit that has bin/nvcc ─────────────────────────────────
find_cuda() {
  local candidates=()
  # Honor caller-provided CUDA_HOME first, but only if it actually has nvcc.
  if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "$CUDA_HOME"; return
  fi
  # Try common versioned installs, newest first.
  for v in 12.8 12.6 12.4 12.1 12.0; do
    candidates+=("/usr/local/cuda-${v}")
  done
  # Generic symlink + any other /usr/local/cuda-* present.
  candidates+=("/usr/local/cuda")
  while IFS= read -r d; do candidates+=("$d"); done < <(ls -d /usr/local/cuda-* 2>/dev/null | sort -Vr)

  for c in "${candidates[@]}"; do
    if [ -x "${c}/bin/nvcc" ]; then
      echo "$c"; return
    fi
  done

  # Last resort: derive from `which nvcc`.
  local nvcc_path
  nvcc_path="$(command -v nvcc || true)"
  if [ -n "$nvcc_path" ]; then
    echo "$(dirname "$(dirname "$nvcc_path")")"; return
  fi

  echo ""
}

CUDA_HOME="$(find_cuda)"
if [ -z "$CUDA_HOME" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "ERROR: could not find a CUDA toolkit with bin/nvcc." >&2
  echo "  Tried: \$CUDA_HOME, /usr/local/cuda-12.{8,6,4,1,0}, /usr/local/cuda, and which nvcc." >&2
  echo "  Install CUDA 12.8+ (Blackwell needs >= 12.8) or export CUDA_HOME=/path/to/cuda." >&2
  exit 1
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

NVCC_VER="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
echo "=== kzin build ==="
echo "CUDA_HOME : ${CUDA_HOME}"
echo "nvcc      : ${CUDA_HOME}/bin/nvcc (${NVCC_VER})"

# ── 2. LibTorch path ─────────────────────────────────────────────────────────
LIBTORCH_DIR="${LIBTORCH_DIR:-/opt/libtorch}"
if [ ! -d "$LIBTORCH_DIR" ]; then
  echo "ERROR: LibTorch not found at ${LIBTORCH_DIR}." >&2
  echo "  Set LIBTORCH_DIR=/path/to/libtorch or install it under /opt/libtorch." >&2
  exit 1
fi
echo "LibTorch  : ${LIBTORCH_DIR}"

# ── 3. CUDA arch list (Blackwell sm_120 for 5060 Ti) ─────────────────────────
# Keep Ampere/Ada/Hopper too so the same build runs elsewhere; trim if you want.
ARCHES="${CMAKE_CUDA_ARCHITECTURES:-86;89;90;120}"
echo "Arches    : ${ARCHES}"
echo

# ── 4. Configure ─────────────────────────────────────────────────────────────
cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCHES}" \
  "$@"

# ── 5. Build ─────────────────────────────────────────────────────────────────
NPROC="$(nproc 2>/dev/null || echo 4)"
cmake --build "$BUILD_DIR" -j "$NPROC"

echo
echo "=== Build complete: ${BUILD_DIR} ==="
