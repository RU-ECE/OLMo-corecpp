/**
 * src/model/attention_config.cpp
 *
 * Thin shim around ATen's scaled_dot_product_attention. Centralises
 * the choice of "attention backend" (sdpa, flash2, flash3,
 * transformer_engine) so the various attention modules don't each
 * duplicate the dispatch logic.
 *
 * "SDPA" = scaled-dot-product attention. ATen's sdpa op is itself a
 * dispatcher that picks the fastest available implementation under
 * the hood: FlashAttention-2 if the input shapes are supported and
 * the GPU has the right tensor-core arch, mem-efficient attention
 * otherwise, with a math-only fallback as the last resort. We just
 * lean on that and pick a hint via the math/efficient/flash backend
 * flags exposed at the global ATen context.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/attention_config.hpp : compute_attention() decl.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/attention.cpp / fused_attention.cpp:
 *     compute_attention(q, k, v, is_causal, ...) is the call inside
 *     the attention forward pass.
 *
 * --- Role in training pipeline ---
 *   The dispatch boundary between "our attention module" and "the
 *   actual SDPA kernel". On the 3060 quickstart the SDPA fallback
 *   is mem-efficient; FlashAttention-2 needs Ada / Hopper.
 */
#include "olmo_cpp/model/attention_config.hpp"
#include <ATen/ops/scaled_dot_product_attention.h>

namespace olmo_cpp {

torch::Tensor compute_attention(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    bool is_causal,
    const SlidingWindowConfig& sw_config,
    double dropout_p) {
  if (!sw_config.is_enabled()) {
    // Full attention via PyTorch SDPA
    return at::scaled_dot_product_attention(
        q, k, v, /*attn_mask=*/c10::nullopt, dropout_p, is_causal);
  }

  // Sliding window attention: create a band mask
  auto S_q = q.size(2);
  auto S_kv = k.size(2);
  auto opts = torch::TensorOptions().dtype(torch::kBool).device(q.device());

  // Build [S_q, S_kv] boolean mask: true = attend, false = masked
  auto row_idx = torch::arange(S_q, torch::TensorOptions().device(q.device())).unsqueeze(1);  // [S_q, 1]
  auto col_idx = torch::arange(S_kv, torch::TensorOptions().device(q.device())).unsqueeze(0); // [1, S_kv]

  // Band mask: position i can attend to [max(0, i - window_size + 1), i]
  // In terms of KV offset when S_q < S_kv (e.g., with cache):
  //   kv_offset = S_kv - S_q
  //   effective_row = row_idx + kv_offset
  auto kv_offset = S_kv - S_q;
  auto effective_row = row_idx + kv_offset;

  // Causal: col <= effective_row
  // Window: col >= effective_row - window_size + 1
  auto mask = (col_idx <= effective_row) & (col_idx >= effective_row - sw_config.window_size + 1);

  // Convert bool mask to float mask for SDPA: 0 where attend, -inf where masked
  auto float_mask = torch::zeros({S_q, S_kv}, torch::TensorOptions().dtype(q.dtype()).device(q.device()));
  float_mask.masked_fill_(~mask, -std::numeric_limits<float>::infinity());

  return at::scaled_dot_product_attention(
      q, k, v, float_mask, dropout_p, /*is_causal=*/false);
}

torch::Tensor sdpa_suffix_causal(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t window,
    torch::Tensor& mask, int64_t& mask_S, int64_t& mask_full_S) {
  const int64_t S = q.size(2);
  const int64_t full_S = k.size(2);
  TORCH_CHECK(S <= full_S, "sdpa_suffix_causal: ", S, " queries but only ", full_S, " keys");
  if (S == 1 && window <= 0) {
    // One query at the end sees every key.
    return at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, false);
  }
  if (S == full_S && window <= 0) {
    // Square causal: SDPA's flag is exact here and keeps the fast kernels.
    return at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true);
  }
  const bool stale = !mask.defined() || S != mask_S || full_S != mask_full_S ||
                     mask.scalar_type() != q.scalar_type() || mask.device() != q.device();
  if (stale) {
    const int64_t offset = full_S - S;
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(q.device());
    auto ones = torch::ones({S, full_S}, bool_opts);
    auto allowed = torch::tril(ones, offset);
    if (window > 0) allowed = allowed & torch::triu(ones, offset - window);
    // Additive mask in the activation dtype (SDPA rejects a fp32 bias on bf16 q/k/v).
    auto mask_opts = torch::TensorOptions().dtype(q.scalar_type()).device(q.device());
    mask = torch::where(allowed, torch::zeros({}, mask_opts),
                        torch::full({}, -std::numeric_limits<float>::infinity(), mask_opts));
    mask_S = S;
    mask_full_S = full_S;
  }
  return at::scaled_dot_product_attention(q, k, v, mask, 0.0, false);
}

}  // namespace olmo_cpp
