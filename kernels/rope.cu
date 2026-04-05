// Fused RoPE CUDA kernels — FP32 and BF16
// 1. apply_rope: Single tensor RoPE
// 2. apply_rope_qk: Fused Q+K RoPE in single kernel (halves launch overhead)
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

template <typename T>
__global__ void apply_rope_kernel(
    const T* __restrict__ x,
    const T* __restrict__ cos_buf,
    const T* __restrict__ sin_buf,
    T* __restrict__ out,
    int64_t total_elements,
    int64_t dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t half = dim / 2;

  for (int64_t i = idx; i < total_elements; i += stride) {
    int64_t col = i % dim;
    int64_t row = i / dim;

    float x_val = static_cast<float>(x[i]);
    float x_rot;
    if (col < half) {
      x_rot = -static_cast<float>(x[row * dim + col + half]);
    } else {
      x_rot = static_cast<float>(x[row * dim + col - half]);
    }
    float c = static_cast<float>(cos_buf[col]);
    float s = static_cast<float>(sin_buf[col]);
    out[i] = static_cast<T>(x_val * c + x_rot * s);
  }
}

template <typename T>
__global__ void apply_rope_qk_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ cos_q,
    const T* __restrict__ sin_q,
    const T* __restrict__ cos_k,
    const T* __restrict__ sin_k,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
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
    const T* src = is_q ? q : k;
    T* dst = is_q ? q_out : k_out;
    const T* cos_ptr = is_q ? cos_q : cos_k;
    const T* sin_ptr = is_q ? sin_q : sin_k;

    int64_t col = local_i % dim;
    int64_t row = local_i / dim;

    float x_val = static_cast<float>(src[local_i]);
    float x_rot;
    if (col < half) {
      x_rot = -static_cast<float>(src[row * dim + col + half]);
    } else {
      x_rot = static_cast<float>(src[row * dim + col - half]);
    }
    float c = static_cast<float>(cos_ptr[col]);
    float s = static_cast<float>(sin_ptr[col]);
    dst[local_i] = static_cast<T>(x_val * c + x_rot * s);
  }
}

}  // namespace

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

  if (x.scalar_type() == torch::kFloat32) {
    apply_rope_kernel<float><<<blocks, threads>>>(
        x_c.data_ptr<float>(),
        cos_c.data_ptr<float>(),
        sin_c.data_ptr<float>(),
        out.data_ptr<float>(),
        numel, dim);
  } else if (x.scalar_type() == torch::kBFloat16) {
    apply_rope_kernel<__nv_bfloat16><<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        numel, dim);
  } else {
    TORCH_CHECK(false, "apply_rope CUDA: unsupported dtype (need fp32 or bf16)");
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

  if (q.scalar_type() == torch::kFloat32) {
    apply_rope_qk_kernel<float><<<blocks, threads>>>(
        q_c.data_ptr<float>(), k_c.data_ptr<float>(),
        cos_q_c.data_ptr<float>(), sin_q_c.data_ptr<float>(),
        cos_k_c.data_ptr<float>(), sin_k_c.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(),
        q_total, k_total, dim);
  } else if (q.scalar_type() == torch::kBFloat16) {
    apply_rope_qk_kernel<__nv_bfloat16><<<blocks, threads>>>(
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
    TORCH_CHECK(false, "apply_rope_qk CUDA: unsupported dtype");
  }

  return {q_out, k_out};
}

TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("apply_rope", apply_rope_cuda);
  m.impl("apply_rope_qk", apply_rope_qk_cuda);
}
