/**
 * kernels/flash_decode.cu
 *
 * FlashAttention-2 decode (single-query) kernel — fast-inference [12].
 *
 * One block per (q_head, batch=1). Block streams K and V in tiles
 * across the n_tokens dimension, accumulating Q·K, online-softmax,
 * and weighted V into a shared-memory accumulator.
 *
 * DRAFT. Same online-softmax structure as paged_attention.cu, just
 * indexed against contiguous K/V tensors instead of paged. Real FA-2
 * adds: warp specialization, async copies via TMA on H100, multi-stage
 * pipelining, register tiling. This draft does none of that.
 *
 * Will need GPU iteration on:
 *   - Block-wide score reduction (currently atomicAdd; should be tree)
 *   - Numerical stability over long sequences
 *   - Vectorized loads on K and V (currently scalar)
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <math_constants.h>
#include <cstdint>

#include "olmo_cpp/backend/flash_decode.hpp"

namespace olmo_cpp {

namespace {

constexpr int kThreads = 128;

__global__ void flash_decode_kernel(
    const float* __restrict__ q,    // [n_q_heads, head_dim]
    const float* __restrict__ k,    // [n_tokens, n_kv_heads, head_dim]
    const float* __restrict__ v,    // [n_tokens, n_kv_heads, head_dim]
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int64_t n_tokens,
    float sm_scale,
    float* __restrict__ out         // [n_q_heads, head_dim]
) {
  int q_head = blockIdx.x;
  if (q_head >= n_q_heads) return;
  int kv_head = (n_kv_heads == n_q_heads)
                    ? q_head
                    : (q_head * n_kv_heads / n_q_heads);

  extern __shared__ float shmem[];
  float* sh_q     = shmem;
  float* sh_accum = shmem + head_dim;
  __shared__ float sh_score;

  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    sh_q[i] = q[q_head * head_dim + i];
    sh_accum[i] = 0.0f;
  }
  __syncthreads();

  float m = -CUDART_INF_F;
  float l = 0.0f;

  for (int64_t t = 0; t < n_tokens; ++t) {
    const float* k_row = k + (t * n_kv_heads + kv_head) * head_dim;
    const float* v_row = v + (t * n_kv_heads + kv_head) * head_dim;

    float score = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      score += sh_q[i] * k_row[i];
    }
    if (threadIdx.x == 0) sh_score = 0.0f;
    __syncthreads();
    atomicAdd(&sh_score, score);
    __syncthreads();
    float s = sh_score * sm_scale;

    float m_new = fmaxf(m, s);
    float exp_diff = __expf(m - m_new);
    float w        = __expf(s - m_new);

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      sh_accum[i] = sh_accum[i] * exp_diff + v_row[i] * w;
    }
    l = l * exp_diff + w;
    m = m_new;
    __syncthreads();
  }

  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    out[q_head * head_dim + i] = sh_accum[i] * inv_l;
  }
}

}  // namespace

torch::Tensor flash_decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, float sm_scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "flash_decode_cuda: must be CUDA");
  TORCH_CHECK(q.dim() == 2 && k.dim() == 3 && v.dim() == 3);

  c10::cuda::CUDAGuard guard(q.device());
  auto q_c = q.contiguous().to(torch::kFloat32);
  auto k_c = k.contiguous().to(torch::kFloat32);
  auto v_c = v.contiguous().to(torch::kFloat32);

  const int n_q_heads  = static_cast<int>(q_c.size(0));
  const int head_dim   = static_cast<int>(q_c.size(1));
  const int n_kv_heads = static_cast<int>(k_c.size(1));
  const int64_t n_tokens = k_c.size(0);

  auto out = torch::empty({n_q_heads, head_dim}, q_c.options());
  size_t shmem = static_cast<size_t>(2 * head_dim) * sizeof(float);

  flash_decode_kernel<<<n_q_heads, kThreads, shmem>>>(
      q_c.data_ptr<float>(),
      k_c.data_ptr<float>(),
      v_c.data_ptr<float>(),
      n_q_heads, n_kv_heads, head_dim, n_tokens, sm_scale,
      out.data_ptr<float>());
  return out.to(q.dtype());
}

}  // namespace olmo_cpp
