/**
 * include/olmo_cpp/model/paged_kv_cache.hpp
 *
 * Fast-inference roadmap item [1] — interface stub.
 *
 * The current KVCache (kv_cache.hpp) holds K/V as torch::Tensors that get
 * concat()-ed each step. Tensor shape changes every step, which:
 *   - blocks CUDA graph capture (item [2])
 *   - makes batched decoding with variable lengths painful
 *   - re-allocates and copies HBM repeatedly
 *
 * The plan: replace the concat-based store with a paged allocator
 * (vLLM-style PagedAttention). Fixed-size pages (e.g. 16 tokens),
 * a per-request page table, and an attention kernel that gathers from
 * pages via the table.
 *
 * This file is a stub. It defines the interface the rest of the code
 * will eventually call and provides a no-op fallback that delegates to
 * the existing concat KVCache. Real implementation lands in:
 *   - kernels/paged_attention.cu  (the gathered attention kernel)
 *   - src/model/paged_kv_cache.cpp (allocator + page table)
 *
 * Until then this header lets us thread `IPagedKVCache*` through the
 * model and bench code without committing to an implementation. Once
 * the paged allocator lands, only the implementations swap.
 *
 * NOT IMPLEMENTED YET. Do not call from production paths. The bench
 * harness uses the existing KVCache; that's deliberate.
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <memory>
#include <utility>

namespace olmo_cpp {

/// Minimal abstraction over a per-request KV cache. The current
/// concat-based KVCache will satisfy this shim trivially; the paged
/// implementation will satisfy it with O(1) append and zero reshape.
class IPagedKVCache {
 public:
  virtual ~IPagedKVCache() = default;

  /// Number of tokens currently cached for this request.
  virtual int64_t seq_len() const = 0;

  /// Append `k` and `v` (shape [B, T, n_heads, head_dim]) to the layer's cache.
  /// Returns the new total seq_len for that layer.
  virtual int64_t append(int64_t layer, torch::Tensor k, torch::Tensor v) = 0;

  /// Materialize layer-`layer` K and V as contiguous tensors of shape
  /// [B, total_T, n_heads, head_dim] for the existing attention kernel.
  /// Paged impl will return *views* into pages where possible. Concat impl
  /// returns the underlying tensor directly.
  virtual std::pair<torch::Tensor, torch::Tensor> materialize(int64_t layer) const = 0;

  /// Snapshot current length for rollback (used by speculative decoding
  /// at chat.cpp:429 / speculative_decode_step).
  virtual int64_t snapshot() const = 0;

  /// Truncate all layers back to `len`.
  virtual void rollback(int64_t len) = 0;

  /// Drop all cached state.
  virtual void clear() = 0;
};

/// Construct a paged KV cache backed by the existing concat KVCache.
/// Used as the fallback for code paths that need IPagedKVCache but cannot
/// yet take advantage of the paged kernels (e.g. CPU-only build).
std::unique_ptr<IPagedKVCache> make_concat_kv_cache_shim(
    int64_t n_layers, torch::Device device);

/// Construct a real paged KV cache backed by a BlockManager.
///
/// Layout: per-layer K/V pools of shape
///   [max_pages, page_size, n_kv_heads, head_dim].
/// Block (page) allocation happens lazily as logical_len_ crosses page
/// boundaries. The caller is expected to invoke `append` for layers in
/// ascending order each step (layer 0 first); the implementation advances
/// the cursor on layer 0 and back-computes destination slots for the rest.
///
/// `max_pages * page_size` is the hard cap on cached sequence length.
///
/// Currently used as a drop-in storage replacement: `materialize(layer)`
/// returns contiguous [1, n_kv_heads, logical_len, head_dim] views, so the
/// existing SDPA-based attention path works unchanged. Decode-side attention
/// can later dispatch to `paged_attention_decode` (kernels/paged_attention.cu)
/// using the BlockManager pools + page table directly, without materializing.
std::unique_ptr<IPagedKVCache> make_paged_kv_cache(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    int64_t max_pages,
    torch::Device device,
    torch::Dtype dtype);

}  // namespace olmo_cpp
