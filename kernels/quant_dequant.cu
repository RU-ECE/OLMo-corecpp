/**
 * kernels/quant_dequant.cu
 *
 * CUDA dequantization kernels for FP8 E4M3 and INT4 AWQ — fast-inference
 * [14]+[15]. Replaces the CPU scalar loops in src/nn/quant.cpp with
 * device-resident dequant so:
 *   1. Quantized weights can stay on the GPU (no D->H roundtrip).
 *   2. Dequant happens in parallel across the whole [V, H] grid instead
 *      of one element at a time on the host.
 *   3. Future quantized GEMV / GEMM kernels can fuse dequant inline.
 *
 * Dequant rules match src/nn/quant.cpp:
 *   FP8 E4M3: per-tensor scale; single FP32 multiplier.
 *     out[i] = e4m3_to_f32(weight[i]) * scale
 *   INT4 AWQ: per-group scale (group_size along H), two INT4 packed per byte.
 *     out[v, h] = signed_q(byte) * scales[v, h / group_size]
 *     where signed_q = (low or high nibble) - 8.
 *
 * E4M3 layout (1 sign + 4 exp + 3 mantissa):
 *   bits 7        : sign
 *   bits 6..3     : exponent (bias 7)
 *   bits 2..0     : mantissa (3 bits)
 * Special: zero (0x00, 0x80), NaN (S 1111 111 — exp=15 mant=7).
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "olmo_cpp/nn/quant.hpp"

namespace olmo_cpp {

namespace {

// Mirrors the host-side e4m3_to_f32 in src/nn/quant.cpp.
__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
  uint32_t sign = (b >> 7) & 0x1;
  int      exp4 = (b >> 3) & 0xF;
  uint32_t mant = b & 0x7;
  // Zero
  if (exp4 == 0 && mant == 0) {
    uint32_t bits = sign << 31;
    return __int_as_float(static_cast<int>(bits));
  }
  // NaN: exponent all-1 + mantissa all-1 in E4M3 is canonical NaN.
  if (exp4 == 0xF && mant == 0x7) {
    return __int_as_float(0x7fc00000);  // canonical FP32 NaN
  }
  // Subnormals (exp4 == 0, mant != 0): represent as FP32 of value
  //   sign * 2^(-6) * (mant / 8)   since E4M3 bias=7, smallest normal exp=1
  if (exp4 == 0) {
    float v = static_cast<float>(mant) / 8.0f;  // mant/8 in [1/8, 7/8]
    v *= 1.0f / 64.0f;  // 2^-6
    return sign ? -v : v;
  }
  // Normal: 2^(exp4 - 7) * (1 + mant/8)
  int      exp32 = exp4 - 7 + 127;
  uint32_t m32   = mant << 20;
  uint32_t bits  = (sign << 31) | (static_cast<uint32_t>(exp32) << 23) | m32;
  return __int_as_float(static_cast<int>(bits));
}

// FP8 dequant: out[i] = e4m3_to_f32(packed[i]) * scale.
// Grid-strided so the kernel scales to any V*H without launch tuning.
__global__ void dequant_fp8_kernel(const uint8_t* __restrict__ packed,
                                   float scale,
                                   int64_t N,
                                   float* __restrict__ out) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < N;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[i] = e4m3_to_f32(packed[i]) * scale;
  }
}

// INT4 AWQ dequant: each output is one nibble * its group's FP16 scale.
//   group_size : along H (last dim).
//   packed     : [V, H/2] uint8, low nibble = even h, high nibble = odd h.
//   scales     : [V, H/group_size] FP16.
__global__ void dequant_int4_awq_kernel(const uint8_t* __restrict__ packed,
                                        const __nv_bfloat16* __restrict__ scales_fp16, // (we use bf16 for storage; really fp16 in source)
                                        int64_t V,
                                        int64_t H,
                                        int64_t group_size,
                                        float* __restrict__ out) {
  const int64_t total = V * H;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t v  = i / H;
    const int64_t h  = i - v * H;
    const int64_t bi = v * (H / 2) + (h >> 1);
    const uint8_t byte = packed[bi];
    const int     q   = (h & 1) ? (byte >> 4) : (byte & 0x0F);
    const int     sq  = q - 8;  // unpack to [-8, 7]
    const int64_t si = v * (H / group_size) + (h / group_size);
    out[i] = static_cast<float>(sq) * __bfloat162float(scales_fp16[si]);
  }
}

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

inline int grid_for(int64_t n) {
  int64_t blocks = (n + kThreads - 1) / kThreads;
  if (blocks < 1) blocks = 1;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  return static_cast<int>(blocks);
}

}  // namespace

torch::Tensor dequantize_fp8_cuda(const Fp8Quantized& q) {
  TORCH_CHECK(q.weight.is_cuda(), "dequantize_fp8_cuda: weight must be CUDA");
  c10::cuda::CUDAGuard guard(q.weight.device());

  auto packed = q.weight.contiguous();
  const float scale = q.scale.item<float>();
  const int64_t N = packed.numel();

  auto out = torch::empty(packed.sizes(),
      torch::TensorOptions().dtype(torch::kFloat32).device(q.weight.device()));
  dequant_fp8_kernel<<<grid_for(N), kThreads>>>(
      packed.data_ptr<uint8_t>(), scale, N, out.data_ptr<float>());
  return out;
}

torch::Tensor dequantize_int4_awq_cuda(const Int4Quantized& q) {
  TORCH_CHECK(q.weight.is_cuda(), "dequantize_int4_awq_cuda: weight must be CUDA");
  c10::cuda::CUDAGuard guard(q.weight.device());

  auto packed = q.weight.contiguous();
  // Promote scales to BF16 for kernel-friendly read regardless of stored
  // dtype. Using BF16 instead of FP16 inside the kernel because the
  // conversion intrinsic (__bfloat162float) is faster than __half2float on
  // SM_80+. The 2-bit difference in mantissa precision is irrelevant for a
  // weight scale.
  auto scales_bf = q.scales.contiguous().to(torch::kBFloat16);

  const int64_t V = packed.size(0);
  const int64_t H = packed.size(1) * 2;
  const int64_t total = V * H;

  auto out = torch::empty({V, H},
      torch::TensorOptions().dtype(torch::kFloat32).device(q.weight.device()));
  dequant_int4_awq_kernel<<<grid_for(total), kThreads>>>(
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(scales_bf.data_ptr<at::BFloat16>()),
      V, H, q.group_size,
      out.data_ptr<float>());
  return out;
}

}  // namespace olmo_cpp
