#include "olmo_cpp/model/fused_block.hpp"
#include "olmo_cpp/backend/backend.hpp"

namespace olmo_cpp {

FusedTransformerBlockImpl::FusedTransformerBlockImpl(
    const TransformerConfig& cfg, int64_t block_idx)
    : attention_(register_module("attention", FusedAttention(cfg, block_idx))),
      attention_norm_(register_module("attention_norm", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_norm_(register_module("feed_forward_norm", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_(register_module("feed_forward",
          FeedForward(cfg.d_model, cfg.get_hidden_size(), /*bias=*/false, /*use_fused_gate_up=*/true))) {}

torch::Tensor FusedTransformerBlockImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  auto& backend = get_backend();
  backend.begin_scope();

  // Attention with fused QKV projection
  auto h = x + attention_norm_(attention_(x, rope_bufs, start_pos, layer_cache));

  // Feed-forward with fused gate_up projection
  auto out = h + feed_forward_norm_(feed_forward_(h));

  backend.end_scope();
  return out;
}

}  // namespace olmo_cpp
