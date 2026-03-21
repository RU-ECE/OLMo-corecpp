#pragma once
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/attention.hpp"
#include "olmo_cpp/model/feed_forward.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/rope.hpp"
#include "olmo_cpp/model/moe/moe.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// PeriNorm block: applies normalization periodically
/// h = norm(x + attn(x)), out = norm(h + ff(h))
class PeriNormBlockImpl : public torch::nn::Module {
 public:
  PeriNormBlockImpl(const TransformerConfig& cfg, int64_t block_idx);
  torch::Tensor forward(torch::Tensor x, const RoPEBuffers* rope_bufs = nullptr,
                        std::optional<int64_t> start_pos = std::nullopt,
                        LayerKVCache* layer_cache = nullptr);
 private:
  Attention attention_;
  RMSNorm norm1_, norm2_;
  FeedForward feed_forward_;
};
TORCH_MODULE(PeriNormBlock);

/// LayerNormScaled block: pre-norm with learned scaling factor
/// h = x + scale_attn * norm1(attn(x)), out = h + scale_ff * norm2(ff(h))
class LayerNormScaledBlockImpl : public torch::nn::Module {
 public:
  LayerNormScaledBlockImpl(const TransformerConfig& cfg, int64_t block_idx);
  torch::Tensor forward(torch::Tensor x, const RoPEBuffers* rope_bufs = nullptr,
                        std::optional<int64_t> start_pos = std::nullopt,
                        LayerKVCache* layer_cache = nullptr);
 private:
  Attention attention_;
  RMSNorm norm1_, norm2_;
  FeedForward feed_forward_;
  torch::Tensor scale_attn_, scale_ff_;
};
TORCH_MODULE(LayerNormScaledBlock);

/// nGPT Normalized block: all vectors live on the unit hypersphere
/// Normalizes all intermediate representations to unit norm
class NormalizedBlockImpl : public torch::nn::Module {
 public:
  NormalizedBlockImpl(const TransformerConfig& cfg, int64_t block_idx);
  torch::Tensor forward(torch::Tensor x, const RoPEBuffers* rope_bufs = nullptr,
                        std::optional<int64_t> start_pos = std::nullopt,
                        LayerKVCache* layer_cache = nullptr);
 private:
  Attention attention_;
  FeedForward feed_forward_;
  torch::Tensor alpha_attn_, alpha_ff_;  // learnable interpolation factors
  int64_t d_model_;
  torch::Tensor normalize(torch::Tensor x);
};
TORCH_MODULE(NormalizedBlock);

/// MoE Reordered Norm block: replaces FFN with MoE layer
class MoEReorderedNormBlockImpl : public torch::nn::Module {
 public:
  MoEReorderedNormBlockImpl(const TransformerConfig& cfg, int64_t block_idx);
  struct BlockOutput {
    torch::Tensor hidden_states;
    torch::Tensor aux_loss;
  };
  BlockOutput forward(torch::Tensor x, const RoPEBuffers* rope_bufs = nullptr,
                      std::optional<int64_t> start_pos = std::nullopt,
                      LayerKVCache* layer_cache = nullptr);
 private:
  Attention attention_;
  RMSNorm attention_norm_, moe_norm_;
  MoELayer moe_;
  TransformerConfig config_;
};
TORCH_MODULE(MoEReorderedNormBlock);

/// MoE Hybrid block: alternates between dense FFN and MoE
class MoEHybridReorderedNormBlockImpl : public torch::nn::Module {
 public:
  MoEHybridReorderedNormBlockImpl(const TransformerConfig& cfg, int64_t block_idx);
  struct BlockOutput {
    torch::Tensor hidden_states;
    torch::Tensor aux_loss;
  };
  BlockOutput forward(torch::Tensor x, const RoPEBuffers* rope_bufs = nullptr,
                      std::optional<int64_t> start_pos = std::nullopt,
                      LayerKVCache* layer_cache = nullptr);
 private:
  Attention attention_;
  RMSNorm attention_norm_, ff_norm_;
  bool use_moe_;
  std::optional<MoELayer> moe_;
  std::optional<FeedForward> feed_forward_;
  TransformerConfig config_;
};
TORCH_MODULE(MoEHybridReorderedNormBlock);

}  // namespace olmo_cpp
