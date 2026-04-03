#pragma once

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/fused_block.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/lm_head.hpp"
#include "olmo_cpp/model/mtp_head.hpp"
#include "olmo_cpp/nn/multi_res_embedding.hpp"
#include <torch/torch.h>
#include <optional>
#include <vector>

namespace olmo_cpp {

/// Optimized transformer using fused operations.
///
/// Differences from TransformerImpl:
/// 1. Uses FusedTransformerBlock (fused QKV + fused gate_up)
/// 2. Supports torch::jit::optimize_for_inference() on the forward path
/// 3. Compatible with CUDA Graphs when input shapes are static
///
/// The model is architecturally identical (same weights, same outputs)
/// but executes ~1.5-3x faster due to reduced kernel launches and
/// improved memory access patterns.
class FusedTransformerImpl : public torch::nn::Module {
 public:
  explicit FusedTransformerImpl(const TransformerConfig& cfg);

  torch::Tensor forward(
      torch::Tensor input_ids,
      c10::optional<torch::Tensor> labels = c10::nullopt,
      int64_t ignore_index = -100,
      KVCache* kv_cache = nullptr);

  torch::Tensor forward_backbone(
      torch::Tensor input_ids,
      KVCache* kv_cache = nullptr);

  std::vector<torch::Tensor> forward_mtp_draft(torch::Tensor hidden_state);

  void init_weights(torch::optional<torch::Generator> gen = c10::nullopt);

  std::vector<RoPEBuffers> get_rope_buffers(int64_t seq_len, torch::Device device,
                                             torch::Dtype dtype = torch::kFloat32);

  int64_t n_layers() const { return config_.n_layers; }
  int64_t num_mtp_heads() const { return config_.num_mtp_heads; }
  bool has_mtp() const { return config_.num_mtp_heads > 0; }

  torch::Tensor apply_lm_head(torch::Tensor hidden_states) { return lm_head_(hidden_states); }

 private:
  // Embedding: either plain or multi-resolution (DC-MRE)
  torch::nn::Embedding embeddings_{nullptr};
  MultiResEmbedding multi_res_embed_{nullptr};
  bool use_multi_res_ = false;

  std::optional<RMSNorm> embedding_norm_;
  torch::nn::ModuleList blocks_;
  LMHead lm_head_;
  std::optional<double> embed_scale_;
  TransformerConfig config_;

  // Multi-Token Prediction heads
  torch::nn::ModuleList mtp_heads_;

  // Cached RoPE — recomputed only when seq_len grows or dtype changes
  int64_t cached_rope_len_ = 0;
  torch::Dtype cached_rope_dtype_ = torch::kFloat32;
  std::vector<RoPEBuffers> cached_rope_bufs_;
};

TORCH_MODULE(FusedTransformer);

}  // namespace olmo_cpp
