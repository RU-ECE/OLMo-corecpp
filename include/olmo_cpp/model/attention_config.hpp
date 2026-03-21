#pragma once
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// Attention backend abstraction
/// The backend determines which kernel is used for the attention computation
enum class AttentionBackendType { SDPA, FlashAttention2, FlashAttention3, TransformerEngine };

/// Sliding window configuration
struct SlidingWindowConfig {
  int64_t window_size = -1;  // -1 = disabled (full attention)
  bool is_enabled() const { return window_size > 0; }
};

/// Compute attention with optional sliding window
/// q,k,v: [B, H, S, D]
/// Returns: [B, H, S, D]
torch::Tensor compute_attention(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    bool is_causal,
    const SlidingWindowConfig& sw_config = {},
    double dropout_p = 0.0);

}  // namespace olmo_cpp
