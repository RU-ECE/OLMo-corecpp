/**
 * src/model/attention.cpp
 *
 * Implementation of the un-fused multi-head / grouped-query self-attention
 * module (AttentionImpl). Constructs Q, K, V, output linear projections,
 * optional QK-RMSNorm, and RoPE, then computes attention via ATen's
 * scaled_dot_product_attention (which dispatches to FlashAttention on
 * CUDA where supported). Sliding-window masks are built lazily and cached
 * across forwards while their dimensions stay constant.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/attention.hpp: own header (declares AttentionImpl,
 *     RoPE, RMSNorm, KVCache types)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/block.cpp: ReorderedNormTransformerBlockImpl::forward
 *     calls Attention(...) — i.e., this forward — once per layer
 *   - src/model/block_variants.cpp: PeriNorm/LayerNormScaled/Normalized
 *     /MoE block forwards likewise dispatch through Attention(...)
 *
 * --- Role in training pipeline ---
 *   On the un-fused training path, this is the per-layer attention
 *   compute. One call per layer per forward; gradients flow back through
 *   ATen's autograd over the SDPA op and the linear layers.
 */
#include "olmo_cpp/model/attention.hpp"
#include <ATen/ops/scaled_dot_product_attention.h>
#include <limits>

namespace olmo_cpp {

/// Construct Q/K/V/output linears, optional QK-norms, and RoPE module.
/// The K/V projections are sized for n_kv_heads (GQA-aware), while the
/// Q and output projections use the full n_heads count.
AttentionImpl::AttentionImpl(const TransformerConfig& cfg, int64_t /*layer_idx*/)
    : w_q_(register_module("w_q", torch::nn::Linear(torch::nn::LinearOptions(cfg.d_model, cfg.n_heads * cfg.get_head_dim()).bias(false)))),
      w_k_(register_module("w_k", torch::nn::Linear(torch::nn::LinearOptions(cfg.d_model, cfg.get_n_kv_heads() * cfg.get_head_dim()).bias(false)))),
      w_v_(register_module("w_v", torch::nn::Linear(torch::nn::LinearOptions(cfg.d_model, cfg.get_n_kv_heads() * cfg.get_head_dim()).bias(false)))),
      w_out_(register_module("w_out", torch::nn::Linear(torch::nn::LinearOptions(cfg.n_heads * cfg.get_head_dim(), cfg.d_model).bias(false)))),
      n_heads_(cfg.n_heads),
      n_kv_heads_(cfg.get_n_kv_heads()),
      head_dim_(cfg.get_head_dim()),
      n_heads_rep_(cfg.n_heads / cfg.get_n_kv_heads()),
      use_head_qk_norm_(cfg.use_head_qk_norm),
      sliding_window_size_(cfg.sliding_window_size) {
  // Optional QK-norm: stabilizes attention scores at large depths/widths.
  if (cfg.use_qk_norm) {
    if (cfg.use_head_qk_norm) {
      // Per-head RMSNorm: parameter vector has length head_dim.
      q_norm_ = RMSNorm(cfg.get_head_dim(), cfg.layer_norm_eps);
      k_norm_ = RMSNorm(cfg.get_head_dim(), cfg.layer_norm_eps);
    } else {
      // Per-tensor RMSNorm: applied before head reshape over the full
      // n_heads*head_dim (Q) or n_kv_heads*head_dim (K) feature width.
      q_norm_ = RMSNorm(cfg.n_heads * cfg.get_head_dim(), cfg.layer_norm_eps);
      k_norm_ = RMSNorm(cfg.get_n_kv_heads() * cfg.get_head_dim(), cfg.layer_norm_eps);
    }
    // Register so they get serialized with the module hierarchy.
    register_module("q_norm", q_norm_.value());
    register_module("k_norm", k_norm_.value());
  }
  // RoPE module is always created (gated at forward by rope_bufs != null).
  rope_ = RotaryEmbedding(cfg.get_head_dim(), cfg.rope_theta);
  register_module("rope", rope_.value());
}

torch::Tensor AttentionImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  auto B = x.size(0);
  auto S = x.size(1);

  auto q = w_q_(x);
  auto k = w_k_(x);
  auto v = w_v_(x);

  // QK-norm before reshape
  if (q_norm_ && !use_head_qk_norm_) q = (*q_norm_)(q);
  if (k_norm_ && !use_head_qk_norm_) k = (*k_norm_)(k);

  q = q.view({B, S, n_heads_, head_dim_}).transpose(1, 2);
  k = k.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);
  v = v.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);

  // QK-norm per head after reshape
  if (q_norm_ && use_head_qk_norm_) q = (*q_norm_)(q);
  if (k_norm_ && use_head_qk_norm_) k = (*k_norm_)(k);

  if (rope_bufs) {
    auto [q_rot, k_rot] = (*rope_)->apply(q, k, *rope_bufs, start_pos);
    q = q_rot;
    k = k_rot;
  }

  if (layer_cache) {
    auto [full_k, full_v] = layer_cache->update(k, v);
    k = full_k;
    v = full_v;
  }

  if (n_heads_rep_ > 1) {
    // Use expand (view-only, no allocation) instead of repeat_interleave (copies)
    auto kS = k.size(2);
    k = k.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_}).reshape({B, n_heads_, kS, head_dim_});
    v = v.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_}).reshape({B, n_heads_, kS, head_dim_});
  }

  bool is_causal = (S > 1) && (layer_cache == nullptr || layer_cache->seq_len() == S);
  auto full_S = k.size(2);

  torch::Tensor attn_out;

  if (sliding_window_size_ > 0 && S > 1) {
    // Reuse cached mask when dimensions haven't changed.
    if (S != cached_mask_S_ || full_S != cached_mask_full_S_) {
      // The mask we want is the band of diagonals [offset - window, offset]
      // (causal upper bound at offset, lower bound at offset - window). Build
      // it as a single bool kernel via tril/triu so we don't allocate two
      // arange tensors and pile up boolean ops. The previous implementation
      // also had a no-op refinement: the original `mask` already enforced
      // `cols <= (rows + offset)`, so the `if (is_causal)` clause re-ANDed
      // an identical condition for free.
      const auto offset = full_S - S;
      auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(x.device());
      auto ones      = torch::ones({S, full_S}, bool_opts);
      auto allowed   = torch::tril(ones, offset)
                         & torch::triu(ones, offset - sliding_window_size_);

      // Mask dtype must match the activation dtype so SDPA doesn't reject a
      // BF16 q/k/v with an FP32 attention bias (same class of bug as the one
      // that killed 7B startup). torch::where with rank-0 tensors avoids the
      // alloc-of-two-SxS-tensors-then-discard pattern the old code used.
      auto mask_opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());
      cached_attn_mask_ = torch::where(
          allowed,
          torch::zeros({}, mask_opts),
          torch::full({}, -std::numeric_limits<float>::infinity(), mask_opts));
      cached_mask_S_      = S;
      cached_mask_full_S_ = full_S;
    }

    attn_out = at::scaled_dot_product_attention(q, k, v, cached_attn_mask_, 0.0, false);
  } else {
    attn_out = at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, is_causal);
  }

  attn_out = attn_out.transpose(1, 2).reshape({B, S, -1});
  return w_out_(attn_out);
}

// ──────────────────────────────────────────────────────────────────────────
// Paged-KV variant. Differs from `forward` only in cache I/O — the rest of
// the path (projections, QK-norm, RoPE, GQA expand, sliding-window mask,
// SDPA) is identical. Kept as a parallel function so the legacy concat
// path stays untouched.
// ──────────────────────────────────────────────────────────────────────────
torch::Tensor AttentionImpl::forward_paged(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    int64_t start_pos,
    IPagedKVCache* paged,
    int64_t layer_idx) {
  TORCH_CHECK(paged != nullptr, "AttentionImpl::forward_paged: paged is null");

  auto B = x.size(0);
  auto S = x.size(1);

  auto q = w_q_(x);
  auto k = w_k_(x);
  auto v = w_v_(x);

  if (q_norm_ && !use_head_qk_norm_) q = (*q_norm_)(q);
  if (k_norm_ && !use_head_qk_norm_) k = (*k_norm_)(k);

  q = q.view({B, S, n_heads_, head_dim_}).transpose(1, 2);
  k = k.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);
  v = v.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);

  if (q_norm_ && use_head_qk_norm_) q = (*q_norm_)(q);
  if (k_norm_ && use_head_qk_norm_) k = (*k_norm_)(k);

  if (rope_bufs) {
    auto [q_rot, k_rot] = (*rope_)->apply(q, k, *rope_bufs, std::optional<int64_t>(start_pos));
    q = q_rot;
    k = k_rot;
  }

  // Append new K/V into the page pool, then materialize the full cached
  // K/V as contiguous [B, n_kv_heads, total_len, head_dim] views.
  paged->append(layer_idx, k, v);
  auto [full_k, full_v] = paged->materialize(layer_idx);
  k = full_k;
  v = full_v;

  if (n_heads_rep_ > 1) {
    auto kS = k.size(2);
    k = k.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_}).reshape({B, n_heads_, kS, head_dim_});
    v = v.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_}).reshape({B, n_heads_, kS, head_dim_});
  }

  const int64_t total_S = k.size(2);
  // Same is_causal heuristic as forward(): use built-in causal when the new
  // sequence covers the entire cached range (prefill); otherwise rely on the
  // append order (no explicit mask needed for decode).
  bool is_causal = (S > 1) && (total_S == S);

  torch::Tensor attn_out;
  if (sliding_window_size_ > 0 && S > 1) {
    if (S != cached_mask_S_ || total_S != cached_mask_full_S_) {
      const auto offset = total_S - S;
      auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(x.device());
      auto ones      = torch::ones({S, total_S}, bool_opts);
      auto allowed   = torch::tril(ones, offset)
                         & torch::triu(ones, offset - sliding_window_size_);
      auto mask_opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());
      cached_attn_mask_ = torch::where(
          allowed,
          torch::zeros({}, mask_opts),
          torch::full({}, -std::numeric_limits<float>::infinity(), mask_opts));
      cached_mask_S_      = S;
      cached_mask_full_S_ = total_S;
    }
    attn_out = at::scaled_dot_product_attention(q, k, v, cached_attn_mask_, 0.0, false);
  } else {
    attn_out = at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, is_causal);
  }

  attn_out = attn_out.transpose(1, 2).reshape({B, S, -1});
  return w_out_(attn_out);
}

}  // namespace olmo_cpp
