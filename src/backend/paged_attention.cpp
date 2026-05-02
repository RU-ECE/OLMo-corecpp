/**
 * src/backend/paged_attention.cpp
 *
 * CPU reference + dispatch for the paged attention decode kernel
 * (fast-inference [9]).
 *
 * The CPU path gathers K/V via the page table into contiguous buffers,
 * then runs reference attention. Slow but correct — useful for
 * validating the CUDA kernel against ground truth on small inputs.
 *
 * DRAFT.
 */

#include "olmo_cpp/backend/paged_attention.hpp"

#include <torch/torch.h>
#include <cmath>

namespace olmo_cpp {

torch::Tensor paged_attention_decode_cpu(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    int64_t n_tokens,
    float sm_scale) {
  TORCH_CHECK(q.is_cpu() && k_pool.is_cpu() && v_pool.is_cpu() && page_table.is_cpu(),
              "paged_attention_decode_cpu: all tensors must be CPU");

  auto q_c = q.contiguous().to(torch::kFloat32);
  auto k_c = k_pool.contiguous().to(torch::kFloat32);
  auto v_c = v_pool.contiguous().to(torch::kFloat32);
  auto pt  = page_table.contiguous().to(torch::kInt32);

  const int64_t n_q_heads  = q_c.size(0);
  const int64_t head_dim   = q_c.size(1);
  const int64_t max_pages  = k_c.size(0);
  const int64_t page_size  = k_c.size(1);
  const int64_t n_kv_heads = k_c.size(2);

  // Gather K and V into [n_tokens, n_kv_heads, head_dim].
  auto k_gathered = torch::empty({n_tokens, n_kv_heads, head_dim}, k_c.options());
  auto v_gathered = torch::empty({n_tokens, n_kv_heads, head_dim}, v_c.options());

  auto pt_a = pt.accessor<int32_t, 1>();
  for (int64_t t = 0; t < n_tokens; ++t) {
    int64_t bi  = t / page_size;
    int64_t off = t % page_size;
    int32_t pg  = pt_a[bi];
    TORCH_CHECK(pg >= 0 && pg < max_pages, "page_table out of range");
    k_gathered[t].copy_(k_c[pg][off]);
    v_gathered[t].copy_(v_c[pg][off]);
  }

  // Standard attention: scores = (Q @ K^T) * sm_scale, softmax, @ V.
  // GQA: expand kv heads to n_q_heads if needed.
  if (n_kv_heads != n_q_heads) {
    int64_t group = n_q_heads / n_kv_heads;
    k_gathered = k_gathered.repeat_interleave(group, /*dim=*/1);
    v_gathered = v_gathered.repeat_interleave(group, /*dim=*/1);
  }

  // q: [Hq, D] → [Hq, 1, D]; k_gathered: [T, Hq, D] → [Hq, T, D]; same for v.
  auto q_b = q_c.unsqueeze(1);                              // [Hq, 1, D]
  auto k_b = k_gathered.transpose(0, 1);                    // [Hq, T, D]
  auto v_b = v_gathered.transpose(0, 1);                    // [Hq, T, D]

  auto scores = torch::matmul(q_b, k_b.transpose(-1, -2)) * sm_scale;  // [Hq, 1, T]
  auto attn   = torch::softmax(scores, -1);                            // [Hq, 1, T]
  auto out    = torch::matmul(attn, v_b).squeeze(1);                    // [Hq, D]

  return out.to(q.dtype());
}

torch::Tensor paged_attention_decode(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    int64_t n_tokens,
    float sm_scale) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (q.is_cuda()) {
    return paged_attention_decode_cuda(q, k_pool, v_pool, page_table, n_tokens, sm_scale);
  }
#endif
  return paged_attention_decode_cpu(q, k_pool, v_pool, page_table, n_tokens, sm_scale);
}

}  // namespace olmo_cpp
