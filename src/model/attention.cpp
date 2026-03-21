#include "olmo_cpp/model/attention.hpp"
#include <ATen/ops/scaled_dot_product_attention.h>
#include <limits>

namespace olmo_cpp {

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
  if (cfg.use_qk_norm) {
    if (cfg.use_head_qk_norm) {
      q_norm_ = RMSNorm(cfg.get_head_dim(), cfg.layer_norm_eps);
      k_norm_ = RMSNorm(cfg.get_head_dim(), cfg.layer_norm_eps);
    } else {
      q_norm_ = RMSNorm(cfg.n_heads * cfg.get_head_dim(), cfg.layer_norm_eps);
      k_norm_ = RMSNorm(cfg.get_n_kv_heads() * cfg.get_head_dim(), cfg.layer_norm_eps);
    }
    register_module("q_norm", q_norm_.value());
    register_module("k_norm", k_norm_.value());
  }
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
    k = k.repeat_interleave(n_heads_rep_, 1);
    v = v.repeat_interleave(n_heads_rep_, 1);
  }

  bool is_causal = (S > 1) && (layer_cache == nullptr || layer_cache->seq_len() == S);
  auto full_S = k.size(2);

  torch::Tensor attn_out;

  if (sliding_window_size_ > 0 && S > 1) {
    // Build sliding window + causal mask
    auto rows = torch::arange(S, x.options().dtype(torch::kLong)).unsqueeze(1);
    auto cols = torch::arange(full_S, x.options().dtype(torch::kLong)).unsqueeze(0);
    auto offset = full_S - S;

    // Attend only within window: j >= (i + offset - window_size) and j <= (i + offset)
    auto mask = (cols >= (rows + offset - sliding_window_size_)) & (cols <= (rows + offset));
    if (is_causal) {
      mask = mask & (cols <= (rows + offset));
    }

    auto attn_mask = torch::where(
        mask,
        torch::zeros({S, full_S}, x.options().dtype(torch::kFloat)),
        torch::full({S, full_S}, -std::numeric_limits<float>::infinity(),
                    x.options().dtype(torch::kFloat)));

    attn_out = at::scaled_dot_product_attention(q, k, v, attn_mask, 0.0, false);
  } else {
    attn_out = at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, is_causal);
  }

  attn_out = attn_out.transpose(1, 2).reshape({B, S, -1});
  return w_out_(attn_out);
}

}  // namespace olmo_cpp
