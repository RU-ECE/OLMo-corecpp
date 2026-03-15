#pragma once

#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

/// Per-layer KV cache for incremental autoregressive decoding.
/// Stores accumulated key and value tensors shaped (B, n_kv_heads, cached_len, head_dim).
struct LayerKVCache {
  torch::Tensor k;  // (B, n_kv_heads, cached_len, head_dim)
  torch::Tensor v;  // (B, n_kv_heads, cached_len, head_dim)

  /// Current cached sequence length.
  int64_t seq_len() const { return k.defined() ? k.size(2) : 0; }

  /// Append new K/V and return the full accumulated tensors.
  std::pair<torch::Tensor, torch::Tensor> update(
      torch::Tensor new_k, torch::Tensor new_v) {
    if (!k.defined()) {
      k = new_k;
      v = new_v;
    } else {
      k = torch::cat({k, new_k}, /*dim=*/2);
      v = torch::cat({v, new_v}, /*dim=*/2);
    }
    return {k, v};
  }
};

/// Full model KV cache — one LayerKVCache per transformer layer.
struct KVCache {
  std::vector<LayerKVCache> layers;

  explicit KVCache(int64_t n_layers) : layers(static_cast<size_t>(n_layers)) {}

  /// Current cached sequence length (same across all layers).
  int64_t seq_len() const {
    return layers.empty() ? 0 : layers[0].seq_len();
  }

  /// Reset all cached state.
  void clear() {
    for (auto& l : layers) {
      l.k = torch::Tensor();
      l.v = torch::Tensor();
    }
  }
};

}  // namespace olmo_cpp
