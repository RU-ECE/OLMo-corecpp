// Fused RoPE CUDA kernels — FP32 and BF16
// BF16: reads __nv_bfloat16, rotates in FP32, writes __nv_bfloat16
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

// ---- FP32 single-tensor RoPE ----

__global__ void apply_rope_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ cos_buf,
    const float* __restrict__ sin_buf,
    float* __restrict__ out,
    int64_t total_elements,
    int64_t dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t half = dim / 2;

  for (int64_t i = idx; i < total_elements; i += stride) {
    int64_t col = i % dim;
    int64_t row = i / dim;
    float x_val = x[i];
    float x_rot;
    if (col < half) {
      x_rot = -x[row * dim + col + half];
    } else {
      x_rot = x[row * dim + col - half];
    }
    out[i] = x_val * cos_buf[col] + x_rot * sin_buf[col];
  }
}

// ---- BF16 single-tensor RoPE with FP32 compute ----

__global__ void apply_rope_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_buf,
    const __nv_bfloat16* __restrict__ sin_buf,
    __nv_bfloat16* __restrict__ out,
    int64_t total_elements,
    int64_t dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t half = dim / 2;

  for (int64_t i = idx; i < total_elements; i += stride) {
    int64_t col = i % dim;
    int64_t row = i / dim;
    float x_val = __bfloat162float(x[i]);
    float x_rot;
    if (col < half) {
      x_rot = -__bfloat162float(x[row * dim + col + half]);
    } else {
      x_rot = __bfloat162float(x[row * dim + col - half]);
    }
    float c = __bfloat162float(cos_buf[col]);
    float s = __bfloat162float(sin_buf[col]);
    out[i] = __float2bfloat16(x_val * c + x_rot * s);
  }
}

// ---- FP32 fused Q+K RoPE ----

__global__ void apply_rope_qk_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos_q,
    const float* __restrict__ sin_q,
    const float* __restrict__ cos_k,
    const float* __restrict__ sin_k,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int64_t q_total,
    int64_t k_total,
    int64_t dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t half = dim / 2;
  int64_t combined_total = q_total + k_total;

  for (int64_t i = idx; i < combined_total; i += stride) {
    bool is_q = (i < q_total);
    int64_t local_i = is_q ? i : (i - q_total);
    const float* src = is_q ? q : k;
    float* dst = is_q ? q_out : k_out;
    const float* cos_ptr = is_q ? cos_q : cos_k;
    const float* sin_ptr = is_q ? sin_q : sin_k;

    int64_t col = local_i % dim;
    int64_t row = local_i / dim;

    float x_val = src[local_i];
    float x_rot;
    if (col < half) {
      x_rot = -src[row * dim + col + half];
    } else {
      x_rot = src[row * dim + col - half];
    }
    dst[local_i] = x_val * cos_ptr[col] + x_rot * sin_ptr[col];
  }
}

// ---- BF16 fused Q+K RoPE ----

__global__ void apply_rope_qk_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos_q,
    const __nv_bfloat16* __restrict__ sin_q,
    const __nv_bfloat16* __restrict__ cos_k,
    const __nv_bfloat16* __restrict__ sin_k,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int64_t q_total,
    int64_t k_total,
    int64_t dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t half = dim / 2;
  int64_t combined_total = q_total + k_total;

  for (int64_t i = idx; i < combined_total; i += stride) {
    bool is_q = (i < q_total);
    int64_t local_i = is_q ? i : (i - q_total);
    const __nv_bfloat16* src = is_q ? q : k;
    __nv_bfloat16* dst = is_q ? q_out : k_out;
    const __nv_bfloat16* cos_ptr = is_q ? cos_q : cos_k;
    const __nv_bfloat16* sin_ptr = is_q ? sin_q : sin_k;

    int64_t col = local_i % dim;
    int64_t row = local_i / dim;

    float x_val = __bfloat162float(src[local_i]);
    float x_rot;
    if (col < half) {
      x_rot = -__bfloat162float(src[row * dim + col + half]);
    } else {
      x_rot = __bfloat162float(src[row * dim + col - half]);
    }
    float c = __bfloat162float(cos_ptr[col]);
    float s = __bfloat162float(sin_ptr[col]);
    dst[local_i] = __float2bfloat16(x_val * c + x_rot * s);
  }
}

}  // namespace

// ---- Dispatch ----

torch::Tensor apply_rope_cuda(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda());
  auto x_c = x.contiguous();
  auto cos_c = cos.contiguous();
  auto sin_c = sin.contiguous();
  auto out = torch::empty_like(x_c);
  auto numel = x_c.numel();
  auto dim = x_c.size(-1);
  c10::cuda::CUDAGuard device_guard(x.device());

  int threads = 256;
  int blocks = std::min(static_cast<int64_t>((numel + threads - 1) / threads),
                        static_cast<int64_t>(65535));

  if (x.scalar_type() == torch::kBFloat16) {
    apply_rope_bf16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        numel, dim);
  } else {
    apply_rope_f32_kernel<<<blocks, threads>>>(
        x_c.data_ptr<float>(),
        cos_c.data_ptr<float>(),
        sin_c.data_ptr<float>(),
        out.data_ptr<float>(),
        numel, dim);
  }
  return out;
}

std::vector<torch::Tensor> apply_rope_qk_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos_q,
    const torch::Tensor& sin_q,
    const torch::Tensor& cos_k,
    const torch::Tensor& sin_k) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda());
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto cos_q_c = cos_q.contiguous();
  auto sin_q_c = sin_q.contiguous();
  auto cos_k_c = cos_k.contiguous();
  auto sin_k_c = sin_k.contiguous();

  auto q_out = torch::empty_like(q_c);
  auto k_out = torch::empty_like(k_c);
  auto q_total = q_c.numel();
  auto k_total = k_c.numel();
  auto dim = q_c.size(-1);
  c10::cuda::CUDAGuard device_guard(q.device());

  int threads = 256;
  int64_t combined = q_total + k_total;
  int blocks = std::min((combined + threads - 1) / threads,
                        static_cast<int64_t>(65535));

  if (q.scalar_type() == torch::kBFloat16) {
    apply_rope_qk_bf16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_q_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_q_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_k_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_k_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr<at::BFloat16>()),
        q_total, k_total, dim);
  } else {
    apply_rope_qk_f32_kernel<<<blocks, threads>>>(
        q_c.data_ptr<float>(),
        k_c.data_ptr<float>(),
        cos_q_c.data_ptr<float>(),
        sin_q_c.data_ptr<float>(),
        cos_k_c.data_ptr<float>(),
        sin_k_c.data_ptr<float>(),
        q_out.data_ptr<float>(),
        k_out.data_ptr<float>(),
        q_total, k_total, dim);
  }

  return {q_out, k_out};
}

TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("apply_rope", apply_rope_cuda);
  m.impl("apply_rope_qk", apply_rope_qk_cuda);
}
