#pragma once

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/block.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/lm_head.hpp"
#include "olmo_cpp/model/mtp_head.hpp"
#include "olmo_cpp/nn/multi_res_embedding.hpp"
#include <torch/torch.h>
#include <optional>
#include <vector>

namespace olmo_cpp {

class TransformerImpl : public torch::nn::Module {
 public:
  explicit TransformerImpl(const TransformerConfig& cfg);

  /// Forward pass. When kv_cache is provided, uses incremental decoding:
  /// only the new tokens are processed, and K/V are cached for future steps.
  ///
  /// When labels are provided and MTP heads are active, computes:
  ///   loss = main_ce_loss + mtp_weight * mean(mtp_ce_losses)
  torch::Tensor forward(
      torch::Tensor input_ids,
      c10::optional<torch::Tensor> labels = c10::nullopt,
      int64_t ignore_index = -100,
      KVCache* kv_cache = nullptr);

  /// MTP draft: given hidden state at the last position, returns logits from
  /// each MTP head. Used for speculative decoding.
  /// Returns vector of [vocab_size] logit tensors, one per MTP head.
  std::vector<torch::Tensor> forward_mtp_draft(torch::Tensor hidden_state);

  /// Get the backbone hidden states (no LM head). Used by speculative decode
  /// to verify candidate tokens.
  torch::Tensor forward_backbone(
      torch::Tensor input_ids,
      KVCache* kv_cache = nullptr);

  void init_weights(torch::optional<torch::Generator> gen = c10::nullopt);

  std::vector<RoPEBuffers> get_rope_buffers(int64_t seq_len, torch::Device device,
                                             torch::Dtype dtype = torch::kFloat32);

  int64_t n_layers() const { return config_.n_layers; }
  int64_t num_mtp_heads() const { return config_.num_mtp_heads; }
  bool has_mtp() const { return config_.num_mtp_heads > 0; }

  /// Apply LM head to hidden states → logits
  torch::Tensor apply_lm_head(torch::Tensor hidden_states) { return lm_head_(hidden_states); }

 private:
  // Embedding: either plain or multi-resolution (DC-MRE)
  torch::nn::Embedding embeddings_{nullptr};
  MultiResEmbedding multi_res_embed_{nullptr};  // used when config.use_multi_res
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

TORCH_MODULE(Transformer);

}  // namespace olmo_cpp
