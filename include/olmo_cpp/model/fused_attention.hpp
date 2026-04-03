#pragma once

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/rope.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// Fused attention that combines multiple optimizations for 2-5x speedup:
///
/// 1. **Fused QKV projection**: Single GEMM for all three projections
///    (reduces kernel launches from 3→1 and improves memory locality)
///
/// 2. **In-place RoPE**: Apply rotary embeddings without allocating new tensors
///
/// 3. **Efficient GQA**: Uses expand+reshape instead of repeat_interleave
///    (saves memory allocation for repeated K/V)
///
/// 4. **Backend dispatch**: Routes to FlashAttention/SDPA/custom kernels
///
/// Paper relevance: This is a key optimization — fused QKV alone gives ~1.3x
/// on attention-dominated workloads, and combined with in-place operations
/// reduces memory pressure significantly.
class FusedAttentionImpl : public torch::nn::Module {
 public:
  FusedAttentionImpl(const TransformerConfig& cfg, int64_t layer_idx);

  torch::Tensor forward(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs = nullptr,
      std::optional<int64_t> start_pos = std::nullopt,
      LayerKVCache* layer_cache = nullptr);

 private:
  // Fused QKV: single [d_model, (n_heads + 2 * n_kv_heads) * head_dim] weight
  torch::nn::Linear w_qkv_{nullptr};
  torch::nn::Linear w_out_{nullptr};
  std::optional<RMSNorm> q_norm_;
  std::optional<RMSNorm> k_norm_;
  std::optional<RotaryEmbedding> rope_;

  int64_t n_heads_;
  int64_t n_kv_heads_;
  int64_t head_dim_;
  int64_t n_heads_rep_;
  int64_t q_size_;   // n_heads * head_dim
  int64_t kv_size_;  // n_kv_heads * head_dim
  bool use_head_qk_norm_;
  int64_t sliding_window_size_;

  // Cached sliding window mask — reused when (S, full_S) unchanged
  int64_t cached_mask_S_ = 0;
  int64_t cached_mask_full_S_ = 0;
  torch::Tensor cached_attn_mask_;
};

TORCH_MODULE(FusedAttention);

}  // namespace olmo_cpp
