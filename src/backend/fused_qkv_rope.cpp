/**
 * src/backend/fused_qkv_rope.cpp
 *
 * CPU reference + host dispatcher for fused QKV+RoPE (item G).
 *
 * The CPU path lays out the work as ATen sequential ops (3 splits,
 * 3 reshapes, RoPE apply) — slow but a numerics-correct baseline the
 * CUDA kernel can be validated against bitwise.
 */

#include "olmo_cpp/backend/fused_qkv_rope.hpp"

#include <torch/torch.h>

namespace olmo_cpp {

namespace {

// Apply RoPE in the "interleaved" pairs form: x_even, x_odd ->
//   x_even' = x_even * cos - x_odd * sin
//   x_odd'  = x_even * sin + x_odd * cos
// cos/sin have shape [S, head_dim/2], broadcast over (B, n_heads).
torch::Tensor apply_rope_ref(torch::Tensor t,    // [B, n_heads, S, head_dim]
                              torch::Tensor cos, // [S, head_dim/2]
                              torch::Tensor sin) {
  const int64_t head_dim = t.size(3);
  auto t_view = t.view({t.size(0), t.size(1), t.size(2), head_dim / 2, 2});
  auto x_even = t_view.select(-1, 0);  // [B, H, S, head_dim/2]
  auto x_odd  = t_view.select(-1, 1);
  auto cos_b = cos.view({1, 1, cos.size(0), cos.size(1)});
  auto sin_b = sin.view({1, 1, sin.size(0), sin.size(1)});
  auto y_even = x_even * cos_b - x_odd * sin_b;
  auto y_odd  = x_even * sin_b + x_odd * cos_b;
  auto out = torch::stack({y_even, y_odd}, /*dim=*/-1);
  return out.reshape(t.sizes());
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope_cpu(torch::Tensor x,
                   torch::Tensor w_qkv,
                   torch::Tensor cos,
                   torch::Tensor sin,
                   int64_t n_q_heads,
                   int64_t n_kv_heads,
                   int64_t head_dim) {
  TORCH_CHECK(x.dim() == 3, "x must be [B, S, d]");
  TORCH_CHECK(w_qkv.dim() == 2, "w_qkv must be 2D");
  const int64_t B = x.size(0);
  const int64_t S = x.size(1);
  const int64_t q_dim  = n_q_heads * head_dim;
  const int64_t kv_dim = n_kv_heads * head_dim;
  TORCH_CHECK(w_qkv.size(0) == q_dim + 2 * kv_dim,
              "w_qkv row count must equal (n_q + 2*n_kv) * head_dim");

  auto qkv = torch::nn::functional::linear(x, w_qkv);       // [B, S, q+2kv]
  auto q = qkv.narrow(-1, 0, q_dim);                         // [B, S, q]
  auto k = qkv.narrow(-1, q_dim, kv_dim);                    // [B, S, kv]
  auto v = qkv.narrow(-1, q_dim + kv_dim, kv_dim);

  q = q.view({B, S, n_q_heads,  head_dim}).transpose(1, 2).contiguous();
  k = k.view({B, S, n_kv_heads, head_dim}).transpose(1, 2).contiguous();
  v = v.view({B, S, n_kv_heads, head_dim}).transpose(1, 2).contiguous();

  q = apply_rope_ref(q, cos, sin);
  k = apply_rope_ref(k, cos, sin);
  return {q, k, v};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope(torch::Tensor x,
               torch::Tensor w_qkv,
               torch::Tensor cos,
               torch::Tensor sin,
               int64_t n_q_heads,
               int64_t n_kv_heads,
               int64_t head_dim) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda()) {
    return fused_qkv_rope_cuda(x, w_qkv, cos, sin,
                                n_q_heads, n_kv_heads, head_dim);
  }
#endif
  return fused_qkv_rope_cpu(x, w_qkv, cos, sin,
                             n_q_heads, n_kv_heads, head_dim);
}

}  // namespace olmo_cpp
