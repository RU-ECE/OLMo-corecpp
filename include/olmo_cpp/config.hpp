#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <cmath>

namespace olmo_cpp {

/// Ensure x is rounded up to the nearest multiple of 'of'
inline int64_t ensure_multiple_of(int64_t x, int64_t of) {
  return of * static_cast<int64_t>(std::ceil(static_cast<double>(x) / of));
}

struct TransformerConfig {
  int64_t d_model = 4096;
  int64_t vocab_size = 50257;
  int64_t n_layers = 32;
  int64_t n_heads = 32;
  int64_t n_kv_heads = -1;  // -1 means n_kv_heads = n_heads
  int64_t head_dim = -1;    // -1 means head_dim = d_model / n_heads
  int64_t rope_theta = 500000;
  double layer_norm_eps = 1e-6;
  double init_std = 0.02;
  std::optional<double> embed_scale;
  std::optional<double> embedding_init_std;
  bool use_qk_norm = true;
  bool use_head_qk_norm = false;
  int64_t hidden_size_multiple_of = 256;
  double hidden_size_multiplier = 1.5;

  // === Block variant ===
  enum class BlockType { ReorderedNorm, PeriNorm, NormalizedNGPT, LayerNormScaled, MoEReorderedNorm, MoEHybridReorderedNorm };
  BlockType block_type = BlockType::ReorderedNorm;

  // === Attention backend ===
  enum class AttentionBackend { SDPA, FlashAttention2, FlashAttention3, TransformerEngine };
  AttentionBackend attention_backend = AttentionBackend::SDPA;

  // === Sliding window attention ===
  int64_t sliding_window_size = -1;  // -1 = disabled

  // === Gated attention ===
  enum class GatedAttentionType { None, Headwise, Elementwise };
  GatedAttentionType gated_attention = GatedAttentionType::None;

  // === RoPE scaling ===
  enum class RoPEScalingType { None, ABF, PositionInterpolation, Stepwise, YaRN };
  RoPEScalingType rope_scaling_type = RoPEScalingType::None;
  double rope_scaling_factor = 1.0;
  double rope_yarn_beta_fast = 32.0;
  double rope_yarn_beta_slow = 1.0;

  // === MoE config ===
  bool use_moe = false;
  int64_t moe_num_experts = 8;
  int64_t moe_top_k = 2;
  int64_t moe_hidden_size = -1;
  double moe_capacity_factor = 1.25;
  bool moe_dropless = true;
  double moe_zloss_weight = 1e-3;
  double moe_lb_loss_weight = 1e-2;
  bool moe_hybrid = false;
  int64_t moe_hybrid_interval = 2;

  // === Activation checkpointing ===
  enum class ActivationCheckpointMode { None, Full, SelectedBlocks };
  ActivationCheckpointMode activation_checkpoint_mode = ActivationCheckpointMode::None;
  int64_t activation_checkpoint_interval = 1;

  // === Convolution ===
  bool use_conv = false;
  int64_t conv_kernel_size = 4;

  // === Float8 ===
  bool use_float8 = false;

  // === Layer norm type ===
  enum class LayerNormType { RMSNorm, LayerNorm, L2Norm, FusedRMSNorm };
  LayerNormType layer_norm_type = LayerNormType::RMSNorm;

  /// Compute effective MoE hidden size
  int64_t get_moe_hidden_size() const {
    return moe_hidden_size > 0 ? moe_hidden_size : get_hidden_size();
  }

  /// Compute effective n_kv_heads (defaults to n_heads if not set)
  int64_t get_n_kv_heads() const {
    return n_kv_heads > 0 ? n_kv_heads : n_heads;
  }

  /// Compute effective head_dim (defaults to d_model / n_heads)
  int64_t get_head_dim() const {
    return head_dim > 0 ? head_dim : (d_model / n_heads);
  }

  /// Compute hidden size for feed-forward: 8/3 * d_model * multiplier, rounded
  int64_t get_hidden_size() const {
    int64_t base = static_cast<int64_t>(8 * d_model / 3.0 * hidden_size_multiplier);
    return ensure_multiple_of(base, hidden_size_multiple_of);
  }

  void validate() const;
};

/// Load config from JSON file (requires nlohmann/json)
TransformerConfig load_config_from_json(const std::string& path);

/// Create olmo2_7B preset
TransformerConfig olmo2_7b_config(int64_t vocab_size = 50257);

}  // namespace olmo_cpp
