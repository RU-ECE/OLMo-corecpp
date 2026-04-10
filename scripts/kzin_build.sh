#!/usr/bin/env bash
# Use one toolkit for nvcc + libraries (avoid /usr/bin/nvcc + CUDA 13 ptxas on PATH).
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
exec cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  "$@"
