cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/libtorch -DCUDAToolkit_ROOT=/usr/local/cuda-12.1 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc
