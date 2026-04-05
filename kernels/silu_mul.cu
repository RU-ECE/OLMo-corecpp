// Fused SiLU(gate) * up CUDA kernel
// Supports FP32 and BF16. Eliminates intermediate tensor allocation.
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

// Vectorized FP32 SiLU*Mul
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

// BF16 SiLU*Mul — compute in FP32 for precision, read/write in BF16
__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  // Process 2 BF16 elements at a time via nv_bfloat162
  int64_t vec_n = n / 2;
  const __nv_bfloat162* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
  const __nv_bfloat162* up2 = reinterpret_cast<const __nv_bfloat162*>(up);
  __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

  for (int64_t i = idx; i < vec_n; i += stride) {
    __nv_bfloat162 g = gate2[i];
    __nv_bfloat162 u = up2[i];
    float g0 = __bfloat162float(__low2bfloat16(g));
    float g1 = __bfloat162float(__high2bfloat16(g));
    float u0 = __bfloat162float(__low2bfloat16(u));
    float u1 = __bfloat162float(__high2bfloat16(u));
    float r0 = silu(g0) * u0;
    float r1 = silu(g1) * u1;
    out2[i] = __floats2bfloat162_rn(r0, r1);
  }

  // Handle odd element
  if (n % 2 != 0) {
    int64_t last = n - 1;
    if (idx == 0) {
      float g = __bfloat162float(gate[last]);
      float u = __bfloat162float(up[last]);
      out[last] = __float2bfloat16(silu(g) * u);
    }
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

  if (gate.scalar_type() == torch::kFloat32) {
    silu_mul_f32_kernel<<<blocks, threads>>>(
        gate_c.data_ptr<float>(),
        up_c.data_ptr<float>(),
        out.data_ptr<float>(),
        n);
  } else if (gate.scalar_type() == torch::kBFloat16) {
    silu_mul_bf16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        n);
  } else {
    TORCH_CHECK(false, "silu_mul CUDA: unsupported dtype (need fp32 or bf16)");
  }
  return out;
}

TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("silu_mul", silu_mul_cuda_impl);
}
