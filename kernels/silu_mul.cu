// Fused SiLU(gate) * up CUDA kernel — FP32 and BF16
// BF16: reads __nv_bfloat16, computes SiLU in FP32, writes __nv_bfloat16
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

// ---- FP32 vectorized ----

__global__ void silu_mul_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

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

  for (int64_t i = vec_n * 4 + idx; i < n; i += stride) {
    out[i] = silu(gate[i]) * up[i];
  }
}

// ---- BF16 with FP32 compute ----

__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t i = idx; i < n; i += stride) {
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    out[i] = __float2bfloat16(silu(g) * u);
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

  if (gate.scalar_type() == torch::kBFloat16) {
    silu_mul_bf16_kernel<<<blocks, threads>>>(
        gate_c.data_ptr<at::BFloat16>(),
        up_c.data_ptr<at::BFloat16>(),
        out.data_ptr<at::BFloat16>(),
        n);
  } else {
    silu_mul_f32_kernel<<<blocks, threads>>>(
        gate_c.data_ptr<float>(),
        up_c.data_ptr<float>(),
        out.data_ptr<float>(),
        n);
  }
  return out;
}

TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("silu_mul", silu_mul_cuda_impl);
}
