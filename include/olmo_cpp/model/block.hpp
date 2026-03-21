#pragma once

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/attention.hpp"
#include "olmo_cpp/model/feed_forward.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/rope.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// ReorderedNormTransformerBlock: h = x + attn_norm(attn(x)), out = h + ff_norm(ff(h))
class ReorderedNormTransformerBlockImpl : public torch::nn::Module {
 public:
  ReorderedNormTransformerBlockImpl(const TransformerConfig& cfg, int64_t block_idx);

  torch::Tensor forward(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs = nullptr,
      std::optional<int64_t> start_pos = std::nullopt,
      LayerKVCache* layer_cache = nullptr);

 private:
  Attention attention_;
  RMSNorm attention_norm_;
  RMSNorm feed_forward_norm_;
  FeedForward feed_forward_;
};

TORCH_MODULE(ReorderedNormTransformerBlock);

}  // namespace olmo_cpp
