/**
 * kernels/paged_attention.cu
 *
 * Paged single-query attention kernel — fast-inference [9].
 *
 * DRAFT. Single-query (decode) only. One block per query head; threads
 * within the block walk the page table, compute Q·K for each cached
 * token, online softmax, accumulate weighted V.
 *
 * Algorithm: standard "FlashDecoding" style online softmax over the
 * gathered KV. Per-query reduction across cached tokens.
 *
 * NOT YET WIRED into the Transformer attention forward. That refactor
 * lives elsewhere.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <math_constants.h>
#include <cstdint>
#include <cmath>

#include "olmo_cpp/backend/paged_attention.hpp"

namespace olmo_cpp {

namespace {

constexpr int kThreads = 128;

// One block per (q_head, batch=1). Each block streams over n_tokens
// cached positions, gathering K/V from pages via the page table, and
// online-softmaxes the result. Output one row of [n_q_heads, head_dim].
__global__ void paged_attention_decode_kernel(
    const float* __restrict__ q,           // [n_q_heads, head_dim]
    const float* __restrict__ k_pool,      // [max_pages, page_size, n_kv_heads, head_dim]
    const float* __restrict__ v_pool,      // same
    const int32_t* __restrict__ page_table,// [n_blocks]
    int64_t n_tokens,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int page_size,
    int max_pages,
    float sm_scale,
    float* __restrict__ out                // [n_q_heads, head_dim]
) {
  int q_head = blockIdx.x;
  if (q_head >= n_q_heads) return;

  // GQA: map q_head → kv_head.
  int kv_head = (n_kv_heads == n_q_heads)
                    ? q_head
                    : (q_head * n_kv_heads / n_q_heads);

  extern __shared__ float shmem[];
  float* sh_q     = shmem;                         // [head_dim]
  float* sh_accum = shmem + head_dim;              // [head_dim]

  // Load Q row into shared memory.
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    sh_q[i] = q[q_head * head_dim + i];
    sh_accum[i] = 0.0f;
  }
  __syncthreads();

  // Online softmax stats.
  float m = -CUDART_INF_F;   // running max
  float l = 0.0f;            // running sum

  // Walk every cached token sequentially. Each thread block owns one
  // q_head, so contention across blocks is on KV-pool reads only.
  for (int64_t t = 0; t < n_tokens; ++t) {
    int64_t block_idx = t / page_size;
    int64_t in_block  = t % page_size;
    int32_t pg = page_table[block_idx];
    if (pg < 0 || pg >= max_pages) continue;

    // Compute attention score: dot(Q, K[pg, in_block, kv_head]).
    const float* k_row = k_pool +
        (((static_cast<int64_t>(pg) * page_size) + in_block) * n_kv_heads + kv_head) * head_dim;

    float score = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      score += sh_q[i] * k_row[i];
    }

    // Warp + block reduce score.
    __shared__ float sh_score;
    if (threadIdx.x == 0) sh_score = 0.0f;
    __syncthreads();
    atomicAdd(&sh_score, score);
    __syncthreads();

    float s = sh_score * sm_scale;

    // Online softmax: re-scale running accumulators by exp(m_old - m_new).
    float m_new = fmaxf(m, s);
    float exp_diff = __expf(m - m_new);   // factor for the existing accumulator
    float w        = __expf(s - m_new);   // factor for the new (Q,K) contribution

    const float* v_row = v_pool +
        (((static_cast<int64_t>(pg) * page_size) + in_block) * n_kv_heads + kv_head) * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      sh_accum[i] = sh_accum[i] * exp_diff + v_row[i] * w;
    }
    l = l * exp_diff + w;
    m = m_new;
    __syncthreads();
  }

  // Final normalization: divide by total weight.
  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    out[q_head * head_dim + i] = sh_accum[i] * inv_l;
  }
}

}  // namespace

torch::Tensor paged_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    int64_t n_tokens,
    float sm_scale) {
  TORCH_CHECK(q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda() && page_table.is_cuda(),
              "paged_attention_decode_cuda: all tensors must be CUDA");
  TORCH_CHECK(q.dim() == 2, "q must be [n_q_heads, head_dim]");
  TORCH_CHECK(k_pool.dim() == 4, "k_pool must be [max_pages, page_size, n_kv_heads, head_dim]");
  TORCH_CHECK(v_pool.dim() == 4, "v_pool must be [max_pages, page_size, n_kv_heads, head_dim]");

  c10::cuda::CUDAGuard guard(q.device());

  auto q_c = q.contiguous().to(torch::kFloat32);
  auto k_c = k_pool.contiguous().to(torch::kFloat32);
  auto v_c = v_pool.contiguous().to(torch::kFloat32);
  auto pt  = page_table.contiguous().to(torch::kInt32);

  const int n_q_heads = static_cast<int>(q_c.size(0));
  const int head_dim  = static_cast<int>(q_c.size(1));
  const int max_pages = static_cast<int>(k_c.size(0));
  const int page_size = static_cast<int>(k_c.size(1));
  const int n_kv_heads = static_cast<int>(k_c.size(2));

  auto out = torch::empty({n_q_heads, head_dim}, q_c.options());
  const size_t shmem = static_cast<size_t>(2 * head_dim) * sizeof(float);

  paged_attention_decode_kernel<<<n_q_heads, kThreads, shmem>>>(
      q_c.data_ptr<float>(),
      k_c.data_ptr<float>(),
      v_c.data_ptr<float>(),
      pt.data_ptr<int32_t>(),
      n_tokens, n_q_heads, n_kv_heads, head_dim,
      page_size, max_pages, sm_scale,
      out.data_ptr<float>());

  return out.to(q.dtype());
}

}  // namespace olmo_cpp
