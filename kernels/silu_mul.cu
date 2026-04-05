// Fused SiLU(gate) * up CUDA kernel
// Eliminates intermediate tensor allocation between SiLU and multiply.
// On H100: saves ~2x memory bandwidth (read gate, read up, write output vs
// read gate, write silu_gate, read silu_gate, read up, write output).
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

// Vectorized SiLU*Mul: 4 elements at a time
__global__ void silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  // Process 4 elements at a time
  int64_t vec_n = n / 4;
  const float4* gate4 = reinterpret_cast<const float4*>(gate);
  const float4* up4 = reinterpret_cast<const float4*>(up);
  float4* out4 = reinterpret_cast<float4*>(out);

  for (int64_t i = idx; i < vec_n; i += stride) {
    float4 g = gate4[i];
    float4 u = up4[i];
    float4 r;
    r.x = silu(g.x) * u.x;
    r.y = silu(g.y) * u.y;
    r.z = silu(g.z) * u.z;
    r.w = silu(g.w) * u.w;
    out4[i] = r;
  }

  // Handle tail elements
  for (int64_t i = vec_n * 4 + idx; i < n; i += stride) {
    out[i] = silu(gate[i]) * up[i];
  }
}

}  // namespace

torch::Tensor silu_mul_cuda_impl(
    const torch::Tensor& gate,
    const torch::Tensor& up) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda());
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have same shape");
  auto gate_c = gate.contiguous();
  auto up_c = up.contiguous();
  auto out = torch::empty_like(gate_c);
  auto n = gate_c.numel();
  c10::cuda::CUDAGuard device_guard(gate.device());

  int threads = 256;
  int blocks = std::min(static_cast<int64_t>((n + threads - 1) / threads),
                        static_cast<int64_t>(65535));

  silu_mul_kernel<<<blocks, threads>>>(
      gate_c.data_ptr<float>(),
      up_c.data_ptr<float>(),
      out.data_ptr<float>(),
      n);
  return out;
}

TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("silu_mul", silu_mul_cuda_impl);
}
