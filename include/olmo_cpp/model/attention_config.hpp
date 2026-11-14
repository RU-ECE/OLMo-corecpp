#pragma once

/**
 * include/olmo_cpp/model/attention_config.hpp
 *
 * Lightweight free-function helper for computing scaled dot product
 * attention with an optional sliding-window mask, plus the small enums and
 * structs that describe attention backend choices. Intended as a uniform
 * helper for places (notably tests / fallback paths) that need attention
 * without instantiating the full AttentionImpl module.
 *
 * --- Includes from this project ---
 *   (No project headers — pure torch/optional dependency.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep. The helper is provided for
 *   experimentation and for non-Attention code paths that build q/k/v
 *   tensors directly; AttentionImpl::forward and FusedAttentionImpl::forward
 *   inline their own SDPA call rather than going through compute_attention().
 *
 * --- Role in training pipeline ---
 *   Not on the hot path of the transformer forward. Provides a uniform
 *   way to invoke SDPA + sliding window mask construction outside the
 *   attention modules.
 */

#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// Attention backend abstraction
/// The backend determines which kernel is used for the attention computation
/// (PyTorch SDPA wraps FA2/FA3/Mem-efficient internally; the other entries
/// are reserved for direct dispatch to those libraries when integrated.)
enum class AttentionBackendType { SDPA, FlashAttention2, FlashAttention3, TransformerEngine };

/// Sliding window configuration. window_size is the number of preceding
/// tokens (inclusive of self) that each query position can attend to.
struct SlidingWindowConfig {
  int64_t window_size = -1;  // -1 = disabled (full attention)
  /// True when a finite window is active.
  bool is_enabled() const { return window_size > 0; }
};

/// Compute attention with optional sliding window
/// q,k,v: [B, H, S, D]
/// Returns: [B, H, S, D]
/// is_causal: when true and no window, applies the standard causal mask.
/// sw_config: when enabled, builds a band mask combining causal + window.
/// dropout_p: attention dropout (training only).
torch::Tensor compute_attention(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    bool is_causal,
    const SlidingWindowConfig& sw_config = {},
    double dropout_p = 0.0);

/// Causal SDPA for a query block that sits at the END of the key sequence:
/// training (S_q == S_kv), decode (S_q == 1), and the multi-token cases that
/// append to a KV cache (speculative verify, chunked prefill: 1 < S_q < S_kv).
/// Query i is at absolute position (S_kv - S_q) + i and may see keys
/// j <= (S_kv - S_q) + i. SDPA's own is_causal is top-left aligned when
/// S_q != S_kv, so that case gets an explicit bottom-right mask; before this
/// helper the cached multi-token case ran with no mask at all (bug 5).
/// window > 0 also drops keys j < (S_kv - S_q) + i - window, as the modules
/// always did. The mask is cached in (mask, mask_S, mask_full_S).
torch::Tensor sdpa_suffix_causal(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t window,
    torch::Tensor& mask, int64_t& mask_S, int64_t& mask_full_S);

}  // namespace olmo_cpp
