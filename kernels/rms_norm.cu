// Fused RMSNorm CUDA kernel - optional performance optimization
// Falls back to ATen implementation when not built
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

__global__ void rms_norm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const float* row_x = x + row * dim;
  float* row_out = out + row * dim;

  __shared__ float s_variance;
  float sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = row_x[i];
    sum_sq += v * v;
  }
  sum_sq = blockReduceSum(sum_sq);
  if (threadIdx.x == 0) {
    s_variance = sqrtf(sum_sq / dim + eps);
  }
  __syncthreads();

  float scale = 1.0f / s_variance;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = row_x[i] * scale;
    if (weight) v *= weight[i];
    row_out[i] = v;
  }
}

__device__ __forceinline__ float blockReduceSum(float val) {
  __shared__ float shared[256];
  int tid = threadIdx.x;
  shared[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    __syncthreads();
  }
  return shared[0];
}

}  // namespace

torch::Tensor rms_norm_cuda_impl(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    double eps) {
  TORCH_CHECK(x.dim() >= 2 && x.is_cuda());
  auto dim = x.size(-1);
  auto rows = x.numel() / dim;
  auto out = torch::empty_like(x);
  c10::cuda::CUDAGuard device_guard(x.device());
  rms_norm_fwd_kernel<<<rows, 256>>>(
      x.data_ptr<float>(),
      weight.has_value() ? weight->data_ptr<float>() : nullptr,
      out.data_ptr<float>(),
      dim,
      static_cast<float>(eps));
  return out;
}

TORCH_LIBRARY(olmo_ops, m) {
  m.def("rms_norm(Tensor x, Tensor? weight, float eps=1e-6) -> Tensor");
  m.def("apply_rope(Tensor x, Tensor cos, Tensor sin) -> Tensor");
}

TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("rms_norm", rms_norm_cuda_impl);
}
