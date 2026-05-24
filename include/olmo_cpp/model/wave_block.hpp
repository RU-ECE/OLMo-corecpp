#pragma once

/**
 * include/olmo_cpp/model/wave_block.hpp
 *
 * One transformer layer using a Wave or Hyena sequence mixer instead of
 * standard QK attention. Preserves the OLMo reordered-norm residual pattern:
 *
 *   h   = x + mixer_norm( mixer(x) )
 *   out = h + ff_norm( feed_forward(h) )
 *
 * The sequence mixer is selected at construction time from cfg.sequence_mixer.
 * For Hybrid modes the caller (WaveTransformerImpl) decides per-layer which
 * block type to build; WaveBlockImpl itself always uses the mixer it was given.
 *
 * --- Callers ---
 *   src/model/wave_transformer.cpp : builds one WaveBlock per layer
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/feed_forward.hpp"
#include "olmo_cpp/model/hyena_ssm.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/wave_attention.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

class WaveBlockImpl : public torch::nn::Module {
 public:
  /// use_wave_mixer: if true, use WaveAttention; if false, use HyenaSSM.
  WaveBlockImpl(const TransformerConfig& cfg, int64_t block_idx,
                bool use_wave_mixer);

  /// x: [B, S, d_model] → [B, S, d_model]
  /// rope_bufs / start_pos / layer_cache are accepted but ignored (wave/Hyena
  /// do not use RoPE or KV cache — they process the full sequence at once).
  torch::Tensor forward(torch::Tensor x);

 private:
  bool use_wave_mixer_;

  // Sequence mixer: exactly one of these is non-null
  WaveAttention wave_attn_{nullptr};
  HyenaSSM      hyena_ssm_{nullptr};

  RMSNorm mixer_norm_;
  RMSNorm feed_forward_norm_;
  FeedForward feed_forward_;
};

TORCH_MODULE(WaveBlock);

}  // namespace olmo_cpp
