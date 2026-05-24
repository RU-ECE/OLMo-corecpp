#pragma once

/**
 * include/olmo_cpp/model/wave_transformer.hpp
 *
 * Full language model using Wave / Hyena sequence mixers.
 *
 * Mirrors the standard Transformer interface (same embedding, LM head, and
 * forward signatures) so the training loop in train.cpp / main.cpp can drive
 * it without modification. The only difference is that each block uses a
 * WaveAttention or HyenaSSM mixer instead of standard QK attention.
 *
 * Supported cfg.sequence_mixer values:
 *   Wave        — every layer is WaveAttention (content-modulated wave FFT)
 *   Hyena       — every layer is HyenaSSM (implicit-filter SSM)
 *   HybridWave  — even layers WaveAttention, odd layers standard Attention
 *   HybridHyena — even layers HyenaSSM,     odd layers standard Attention
 *
 * Hybrid modes use a heterogeneous ModuleList: WaveBlock on even indices,
 * ReorderedNormTransformerBlock on odd indices. The extra attention cost on
 * odd layers is roughly halved by the O(n log n) even layers.
 *
 * --- Callers ---
 *   src/main.cpp : instantiated when cfg.sequence_mixer != Attention
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/block.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/lm_head.hpp"
#include "olmo_cpp/model/wave_block.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

class WaveTransformerImpl : public torch::nn::Module {
 public:
  explicit WaveTransformerImpl(const TransformerConfig& cfg);

  /// Training forward: computes cross-entropy loss when labels are given,
  /// otherwise returns logits [B, S, vocab_size].
  torch::Tensor forward(
      torch::Tensor input_ids,
      c10::optional<torch::Tensor> labels = c10::nullopt,
      int64_t ignore_index = -100);

  /// Backbone only: returns hidden states [B, S, d_model]. Used by eval.
  torch::Tensor forward_backbone(torch::Tensor input_ids);

  void init_weights(torch::optional<torch::Generator> gen = c10::nullopt);

  int64_t n_layers() const { return config_.n_layers; }

 private:
  TransformerConfig config_;

  torch::nn::Embedding embeddings_{nullptr};
  std::optional<RMSNorm> embedding_norm_;
  // Heterogeneous block stack: entries are WaveBlock or
  // ReorderedNormTransformerBlock depending on the hybrid pattern.
  torch::nn::ModuleList blocks_;
  LMHead lm_head_;

  // True at index i when that layer is a WaveBlock (vs standard block).
  std::vector<bool> is_wave_layer_;
};

TORCH_MODULE(WaveTransformer);

}  // namespace olmo_cpp
