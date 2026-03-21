#pragma once

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/fused_attention.hpp"
#include "olmo_cpp/model/feed_forward.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/rope.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// Fused transformer block with optimized attention.
///
/// Optimizations over ReorderedNormTransformerBlock:
/// 1. Uses FusedAttention (fused QKV projection, efficient GQA)
/// 2. Fused gate_up in feed-forward (already in FeedForward with flag)
/// 3. Residual connections are written in-place where safe
/// 4. Arena allocator scopes for zero-copy scratch memory
///
/// Architecture: h = x + norm1(fused_attn(x)), out = h + norm2(ffn(h))
class FusedTransformerBlockImpl : public torch::nn::Module {
 public:
  FusedTransformerBlockImpl(const TransformerConfig& cfg, int64_t block_idx);

  torch::Tensor forward(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs = nullptr,
      std::optional<int64_t> start_pos = std::nullopt,
      LayerKVCache* layer_cache = nullptr);

 private:
  FusedAttention attention_;
  RMSNorm attention_norm_;
  RMSNorm feed_forward_norm_;
  FeedForward feed_forward_;
};

TORCH_MODULE(FusedTransformerBlock);

}  // namespace olmo_cpp
