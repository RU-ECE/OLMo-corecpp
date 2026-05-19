#pragma once

/**
 * include/olmo_cpp/model/attention.hpp
 *
 * Multi-head / grouped-query self-attention module used by the baseline
 * (un-fused) Transformer. Wraps four linear projections (Q, K, V, output),
 * optional QK-norm (per-head or per-tensor), rotary positional embeddings
 * (RoPE), and a sliding-window causal mask. Forward dispatches to ATen's
 * scaled_dot_product_attention so on CUDA it transparently uses
 * FlashAttention when available.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp: TransformerConfig (d_model, n_heads, n_kv_heads,
 *     head_dim, RoPE theta, sliding window size, qk-norm flags)
 *   - olmo_cpp/model/kv_cache.hpp: LayerKVCache for incremental decoding
 *   - olmo_cpp/model/layer_norm.hpp: RMSNorm for optional QK-norm
 *   - olmo_cpp/model/rope.hpp: RotaryEmbedding + RoPEBuffers (precomputed
 *     sin/cos tables)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/block.cpp: ReorderedNormTransformerBlockImpl owns one
 *     Attention as the attention sublayer of each block
 *   - src/model/block_variants.cpp: PeriNormBlock, LayerNormScaledBlock,
 *     NormalizedBlock, and the MoE block variants all use this Attention
 *
 * --- Role in training pipeline ---
 *   This is the un-fused reference implementation of self-attention used by
 *   the standard Transformer. It is the unit of attention compute called
 *   once per layer per forward; on the bench path FusedAttention replaces
 *   it but the math is identical.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/paged_kv_cache.hpp"
#include "olmo_cpp/model/rope.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// Standard (un-fused) multi-head self-attention with optional GQA, QK-norm,
/// RoPE and sliding window. Registered with the LibTorch module system via
/// TORCH_MODULE(Attention) below.
class AttentionImpl : public torch::nn::Module {
 public:
  /// Build all linear projections, optional q/k norms, and the RoPE module
  /// from the model config. layer_idx is currently unused but kept for
  /// future per-layer parameter overrides.
  AttentionImpl(const TransformerConfig& cfg, int64_t layer_idx);

  /// Forward pass with optional KV cache for incremental decoding.
  /// x: [B, S, d_model]. Returns [B, S, d_model].
  /// rope_bufs: precomputed sin/cos for this layer (nullptr disables RoPE).
  /// start_pos: token offset of the first row of x (used for cache append).
  /// layer_cache: when non-null, K/V are appended to the cache and the full
  /// cached K/V are used as keys/values.
  torch::Tensor forward(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs = nullptr,
      std::optional<int64_t> start_pos = std::nullopt,
      LayerKVCache* layer_cache = nullptr);

  /// Paged-KV variant of forward. Mirrors `forward` exactly except that the
  /// per-layer K/V append/materialize goes through `paged` at `layer_idx`
  /// instead of a LayerKVCache. Currently still uses ATen SDPA on the
  /// materialized K/V views — the dedicated paged-attention decode kernel
  /// (kernels/paged_attention.cu) can be wired in once dtype handling and
  /// batched-q support are finalised. Decode-only call site; expects
  /// `paged != nullptr`.
  torch::Tensor forward_paged(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs,
      int64_t start_pos,
      IPagedKVCache* paged,
      int64_t layer_idx);

 private:
  /// Q projection: [d_model] -> [n_heads * head_dim].
  torch::nn::Linear w_q_;
  /// K projection: [d_model] -> [n_kv_heads * head_dim] (GQA-aware).
  torch::nn::Linear w_k_;
  /// V projection: [d_model] -> [n_kv_heads * head_dim] (GQA-aware).
  torch::nn::Linear w_v_;
  /// Output projection: [n_heads * head_dim] -> [d_model].
  torch::nn::Linear w_out_;
  /// Optional QK-norm on queries (RMSNorm), per-head or per-tensor.
  std::optional<RMSNorm> q_norm_;
  /// Optional QK-norm on keys (RMSNorm), per-head or per-tensor.
  std::optional<RMSNorm> k_norm_;
  /// Rotary positional embeddings; populated unconditionally in ctor.
  std::optional<RotaryEmbedding> rope_;
  /// Number of query heads (full multi-head count).
  int64_t n_heads_;
  /// Number of key/value heads for grouped-query attention.
  int64_t n_kv_heads_;
  /// Per-head dimensionality (d_model / n_heads in the simple case).
  int64_t head_dim_;
  /// Repetition factor for GQA: each KV head is shared by this many Q heads.
  int64_t n_heads_rep_;  // n_heads / n_kv_heads for GQA repeat
  /// If true, QK-norm is applied per-head after reshape; else pre-reshape.
  bool use_head_qk_norm_;
  /// Sliding window size in tokens; -1 disables (full causal attention).
  int64_t sliding_window_size_;  // -1 = no window (full attention)

  /// Cached query length used to build the last attention mask.
  // Cached sliding window mask — reused when (S, full_S) unchanged
  int64_t cached_mask_S_ = 0;
  /// Cached key length used to build the last attention mask.
  int64_t cached_mask_full_S_ = 0;
  /// Cached additive (-inf / 0) sliding-window mask, dtype-matched to acts.
  torch::Tensor cached_attn_mask_;
};

/// Holder type macro from LibTorch — defines `Attention` as a shared-ptr
/// wrapper around AttentionImpl so it composes cleanly with ModuleList etc.
TORCH_MODULE(Attention);

}  // namespace olmo_cpp
