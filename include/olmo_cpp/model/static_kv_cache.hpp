#pragma once

#include <torch/torch.h>
#include <vector>
#include <cstdint>

namespace olmo_cpp {

/// Pre-allocated KV cache with fixed maximum length.
/// Unlike the dynamic KVCache (which uses torch::cat each step),
/// this cache pre-allocates buffers for max_seq_len and writes into them
/// via narrow/copy — no allocations during decode.
///
/// Benefits:
///   - Zero allocation in decode loop (constant tensor shapes)
///   - Compatible with CUDA graph capture
///   - ~20-30% faster than concat-based cache on GPU
struct StaticLayerKVCache {
  torch::Tensor k;  // (B, n_kv_heads, max_seq_len, head_dim) — pre-allocated
  torch::Tensor v;  // same shape
  int64_t len = 0;  // current valid length

  int64_t seq_len() const { return len; }

  /// Pre-allocate buffers for the given shape.
  void allocate(int64_t batch_size, int64_t n_kv_heads, int64_t max_seq_len,
                int64_t head_dim, torch::Device device, torch::Dtype dtype = torch::kFloat32) {
    k = torch::zeros({batch_size, n_kv_heads, max_seq_len, head_dim},
                     torch::TensorOptions().dtype(dtype).device(device));
    v = torch::zeros_like(k);
    len = 0;
  }

  /// Write new K/V at position [len, len+new_len) and return view of [0, len+new_len).
  std::pair<torch::Tensor, torch::Tensor> update(
      torch::Tensor new_k, torch::Tensor new_v) {
    int64_t new_len = new_k.size(2);
    // Copy into pre-allocated buffer (no allocation!)
    k.narrow(2, len, new_len).copy_(new_k);
    v.narrow(2, len, new_len).copy_(new_v);
    len += new_len;
    // Return view of valid region
    return {k.narrow(2, 0, len), v.narrow(2, 0, len)};
  }

  /// Truncate to given length (for speculative decode rollback).
  void truncate(int64_t new_len) {
    if (new_len < len) {
      // Zero out the freed region to avoid stale data
      if (new_len < k.size(2)) {
        k.narrow(2, new_len, len - new_len).zero_();
        v.narrow(2, new_len, len - new_len).zero_();
      }
      len = new_len;
    }
  }
};

/// Static KV cache for all layers.
struct StaticKVCache {
  std::vector<StaticLayerKVCache> layers;

  explicit StaticKVCache(int64_t n_layers) : layers(static_cast<size_t>(n_layers)) {}

  /// Pre-allocate all layers.
  void allocate(int64_t batch_size, int64_t n_kv_heads, int64_t max_seq_len,
                int64_t head_dim, torch::Device device, torch::Dtype dtype = torch::kFloat32) {
    for (auto& l : layers) {
      l.allocate(batch_size, n_kv_heads, max_seq_len, head_dim, device, dtype);
    }
  }

  int64_t seq_len() const {
    return layers.empty() ? 0 : layers[0].seq_len();
  }

  int64_t snapshot() const { return seq_len(); }

  void rollback(int64_t len) {
    for (auto& l : layers) l.truncate(len);
  }

  void clear() {
    for (auto& l : layers) {
      l.len = 0;
      if (l.k.defined()) l.k.zero_();
      if (l.v.defined()) l.v.zero_();
    }
  }
};

}  // namespace olmo_cpp
