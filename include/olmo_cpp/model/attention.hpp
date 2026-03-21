#pragma once

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/rope.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(const TransformerConfig& cfg, int64_t layer_idx);

  /// Forward pass with optional KV cache for incremental decoding.
  torch::Tensor forward(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs = nullptr,
      std::optional<int64_t> start_pos = std::nullopt,
      LayerKVCache* layer_cache = nullptr);

 private:
  torch::nn::Linear w_q_;
  torch::nn::Linear w_k_;
  torch::nn::Linear w_v_;
  torch::nn::Linear w_out_;
  std::optional<RMSNorm> q_norm_;
  std::optional<RMSNorm> k_norm_;
  std::optional<RotaryEmbedding> rope_;
  int64_t n_heads_;
  int64_t n_kv_heads_;
  int64_t head_dim_;
  int64_t n_heads_rep_;  // n_heads / n_kv_heads for GQA repeat
  bool use_head_qk_norm_;
  int64_t sliding_window_size_;  // -1 = no window (full attention)
};

TORCH_MODULE(Attention);

}  // namespace olmo_cpp
