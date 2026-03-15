// Fused RoPE CUDA kernel - optional performance optimization
// The main RoPE logic uses ATen in rope.cpp; this provides fused apply for hot path
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

// Apply RoPE: out = x * cos + rotate_half(x) * sin
template <typename T>
__global__ void apply_rope_kernel(
    const T* __restrict__ x,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    T* __restrict__ out,
    int64_t dim) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = blockDim.x * gridDim.x;
  int64_t half = dim / 2;

  for (int64_t i = idx; i < total; i += blockDim.x * gridDim.x) {
    int64_t row = i / dim;
    int64_t col = i % dim;
    int64_t d = col % half;
    T x_val = x[i];
    T x_rot = (col < half) ? (-x[row * dim + d + half]) : (x[row * dim + d]);
    out[i] = x_val * cos[col] + x_rot * sin[col];
  }
}

}  // namespace

torch::Tensor apply_rope_cuda(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda());
  auto out = torch::empty_like(x);
  auto numel = x.numel();
  c10::cuda::CUDAGuard device_guard(x.device());
  if (x.scalar_type() == torch::kFloat32) {
    apply_rope_kernel<float><<<(numel + 255) / 256, 256>>>(
        x.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        out.data_ptr<float>(),
        x.size(-1));
  } else {
    TORCH_CHECK(false, "apply_rope CUDA only supports float32");
  }
  return out;
}

TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("apply_rope", apply_rope_cuda);
}
