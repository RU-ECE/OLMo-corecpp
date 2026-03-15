#include "olmo_cpp/model/block.hpp"

namespace olmo_cpp {

ReorderedNormTransformerBlockImpl::ReorderedNormTransformerBlockImpl(
    const TransformerConfig& cfg, int64_t block_idx)
    : attention_(register_module("attention", Attention(cfg, block_idx))),
      attention_norm_(register_module("attention_norm", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_norm_(register_module("feed_forward_norm", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_(register_module("feed_forward", FeedForward(cfg.d_model, cfg.get_hidden_size(), false))) {}

torch::Tensor ReorderedNormTransformerBlockImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  auto h = x + attention_norm_(attention_(x, rope_bufs, start_pos, layer_cache));
  auto out = h + feed_forward_norm_(feed_forward_(h));
  return out;
}

}  // namespace olmo_cpp
