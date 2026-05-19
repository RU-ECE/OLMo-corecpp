/**
 * src/model/paged_kv_cache.cpp
 *
 * Concat-backed implementation of IPagedKVCache — the shim that lets us
 * thread the new interface through the codebase before the real paged
 * allocator (fast-inference [9]) lands.
 *
 * Internally just wraps a KVCache and forwards every call. Append uses
 * KVCache's existing pre-allocated-buffer logic. Materialize returns
 * views over the filled region, identical to what the model code
 * currently sees from KVCache directly.
 *
 * Performance: same as KVCache, no improvement. The point is interface
 * stability — once kernels/paged_attention.cu lands, only this file is
 * replaced; call sites stay put.
 */

#include "olmo_cpp/model/paged_kv_cache.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/block_manager.hpp"

#include <stdexcept>
#include <algorithm>

namespace olmo_cpp {

namespace {

class ConcatKvCacheShim : public IPagedKVCache {
 public:
  ConcatKvCacheShim(int64_t n_layers, torch::Device device)
      : kv_(n_layers, device) {}

  int64_t seq_len() const override { return kv_.seq_len(); }

  int64_t append(int64_t layer, torch::Tensor k, torch::Tensor v) override {
    if (layer < 0 || static_cast<size_t>(layer) >= kv_.layers.size()) {
      throw std::out_of_range("paged_kv_cache_shim: layer index out of range");
    }
    auto& l = kv_.layers[static_cast<size_t>(layer)];
    auto [_, __] = l.update(k, v);
    (void)_; (void)__;
    return l.seq_len();
  }

  std::pair<torch::Tensor, torch::Tensor> materialize(int64_t layer) const override {
    if (layer < 0 || static_cast<size_t>(layer) >= kv_.layers.size()) {
      throw std::out_of_range("paged_kv_cache_shim: layer index out of range");
    }
    const auto& l = kv_.layers[static_cast<size_t>(layer)];
    if (!l.k.defined()) {
      // Empty cache — return empty tensors. Caller should typically check
      // seq_len() before calling materialize on a fresh layer.
      return {torch::Tensor(), torch::Tensor()};
    }
    int64_t n = l.seq_len();
    return {l.k.narrow(2, 0, n), l.v.narrow(2, 0, n)};
  }

  int64_t snapshot() const override { return kv_.snapshot(); }

  void rollback(int64_t len) override { kv_.rollback(len); }

  void clear() override { kv_.clear(); }

  /// Internal accessor for code paths that still take KVCache* directly.
  /// This is the bridge: existing callers continue to work, new callers
  /// use the IPagedKVCache interface.
  KVCache* underlying() { return &kv_; }

 private:
  KVCache kv_;
};

}  // namespace

std::unique_ptr<IPagedKVCache> make_concat_kv_cache_shim(
    int64_t n_layers, torch::Device device) {
  return std::make_unique<ConcatKvCacheShim>(n_layers, device);
}

// ──────────────────────────────────────────────────────────────────────────
// Real paged implementation, backed by BlockManager.
// ──────────────────────────────────────────────────────────────────────────

namespace {

class PagedKVCache : public IPagedKVCache {
 public:
  PagedKVCache(int64_t n_layers,
               int64_t n_kv_heads,
               int64_t head_dim,
               int64_t page_size,
               int64_t max_pages,
               torch::Device device,
               torch::Dtype dtype)
      : mgr_(n_layers, n_kv_heads, head_dim, page_size, max_pages, device, dtype),
        n_layers_(n_layers),
        page_size_(page_size),
        device_(device) {}

  int64_t seq_len() const override { return logical_len_; }

  int64_t append(int64_t layer, torch::Tensor k, torch::Tensor v) override {
    // k, v: [B=1, n_kv_heads, S, head_dim]. We support B==1 only for now.
    TORCH_CHECK(layer >= 0 && layer < n_layers_,
                "PagedKVCache: layer index out of range");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4,
                "PagedKVCache: k/v must be 4D [B, n_kv_heads, S, head_dim]");
    TORCH_CHECK(k.size(0) == 1 && v.size(0) == 1,
                "PagedKVCache: batch>1 not supported on this path");
    TORCH_CHECK(k.sizes() == v.sizes(), "PagedKVCache: k and v must match");
    const int64_t S = k.size(2);
    if (S == 0) return logical_len_;

    // Layer 0 is the "leader": it allocates any new pages and advances the
    // cursor. Layers 1..N-1 just write into the slots that layer 0 reserved.
    int64_t step_start, step_end;
    if (layer == 0) {
      step_start = logical_len_;
      step_end = logical_len_ + S;
      const int64_t blocks_needed = (step_end + page_size_ - 1) / page_size_;
      const int64_t blocks_current = static_cast<int64_t>(mgr_.page_table().size());
      if (blocks_needed > blocks_current) {
        mgr_.allocate(blocks_needed - blocks_current);
      }
      logical_len_ = step_end;
    } else {
      step_end = logical_len_;
      step_start = step_end - S;
      TORCH_CHECK(step_start >= 0,
                  "PagedKVCache: append called for layer>0 before layer 0 advanced cursor");
    }

    write_layer_slots_(layer, k, v, step_start, step_end);
    return logical_len_;
  }

  std::pair<torch::Tensor, torch::Tensor> materialize(int64_t layer) const override {
    TORCH_CHECK(layer >= 0 && layer < n_layers_,
                "PagedKVCache: layer index out of range");
    if (logical_len_ == 0) {
      return {torch::Tensor(), torch::Tensor()};
    }
    return gather_layer_(layer, logical_len_);
  }

  int64_t snapshot() const override { return logical_len_; }

  void rollback(int64_t len) override {
    TORCH_CHECK(len >= 0 && len <= logical_len_,
                "PagedKVCache: rollback length out of range");
    // Pages stay allocated; subsequent appends will overwrite the rolled-back
    // slots. Cheap (just a cursor move) — matches LayerKVCache::truncate.
    logical_len_ = len;
  }

  void clear() override {
    mgr_.free_all();
    logical_len_ = 0;
  }

 private:
  void write_layer_slots_(int64_t layer,
                          torch::Tensor k,
                          torch::Tensor v,
                          int64_t start,
                          int64_t end) {
    // Destination: k_pool[layer][pg, slot, :, :] for each global position in
    // [start, end). Build (pg, slot) index tensors and do one bulk index_put_.
    const int64_t S = end - start;
    const auto& pt = mgr_.page_table();

    // Build host-side index vectors. n_blocks is small (~ ceil(max_seq/16))
    // and S is typically 1 (decode) or prompt_len (prefill); doing this on
    // the host then transferring is cheaper than launching tiny GPU index ops.
    std::vector<int64_t> pg_idx(static_cast<size_t>(S));
    std::vector<int64_t> slot_idx(static_cast<size_t>(S));
    for (int64_t i = 0; i < S; ++i) {
      const int64_t g = start + i;
      const int64_t blk = g / page_size_;
      const int64_t off = g % page_size_;
      pg_idx[static_cast<size_t>(i)]   = static_cast<int64_t>(pt[static_cast<size_t>(blk)]);
      slot_idx[static_cast<size_t>(i)] = off;
    }

    auto opts_cpu = torch::TensorOptions().dtype(torch::kInt64);
    auto pg_t   = torch::from_blob(pg_idx.data(),   {S}, opts_cpu).clone().to(device_);
    auto slot_t = torch::from_blob(slot_idx.data(), {S}, opts_cpu).clone().to(device_);

    // Source: k has shape [1, n_kv_heads, S, head_dim]. We need
    // [S, n_kv_heads, head_dim] to match the pool slice ordering.
    auto k_src = k.select(0, 0).permute({1, 0, 2}).contiguous();  // [S, n_kv_heads, head_dim]
    auto v_src = v.select(0, 0).permute({1, 0, 2}).contiguous();

    auto& k_pool = mgr_.k_pool(layer);
    auto& v_pool = mgr_.v_pool(layer);
    // index_put_ with two index tensors performs gather-style scatter:
    // k_pool[pg_t[i], slot_t[i], :, :] = k_src[i]   for i in [0, S).
    k_pool.index_put_({pg_t, slot_t}, k_src.to(k_pool.dtype()));
    v_pool.index_put_({pg_t, slot_t}, v_src.to(v_pool.dtype()));
  }

  std::pair<torch::Tensor, torch::Tensor> gather_layer_(int64_t layer, int64_t total_len) const {
    // Pool layout: [max_pages, page_size, n_kv_heads, head_dim].
    // Gather the in-use pages, flatten to [n_blocks * page_size, ...], narrow
    // down to the logical length, then reshape to [1, n_kv_heads, total_len, head_dim].
    const auto& pt = mgr_.page_table();
    const int64_t n_blocks = (total_len + page_size_ - 1) / page_size_;
    TORCH_CHECK(n_blocks <= static_cast<int64_t>(pt.size()),
                "PagedKVCache: page table too short for requested length");

    auto opts_cpu = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<int64_t> pt_int64(static_cast<size_t>(n_blocks));
    for (int64_t i = 0; i < n_blocks; ++i) {
      pt_int64[static_cast<size_t>(i)] = static_cast<int64_t>(pt[static_cast<size_t>(i)]);
    }
    auto pt_t = torch::from_blob(pt_int64.data(), {n_blocks}, opts_cpu).clone().to(device_);

    const auto& k_pool = const_cast<BlockManager&>(mgr_).k_pool(layer);
    const auto& v_pool = const_cast<BlockManager&>(mgr_).v_pool(layer);
    const int64_t n_kv_heads = mgr_.n_kv_heads();
    const int64_t head_dim   = mgr_.head_dim();

    auto k_blocks = k_pool.index_select(0, pt_t);  // [n_blocks, page_size, n_kv_heads, head_dim]
    auto v_blocks = v_pool.index_select(0, pt_t);
    auto k_flat = k_blocks.reshape({n_blocks * page_size_, n_kv_heads, head_dim})
                          .narrow(0, 0, total_len);                  // [total_len, H, D]
    auto v_flat = v_blocks.reshape({n_blocks * page_size_, n_kv_heads, head_dim})
                          .narrow(0, 0, total_len);
    auto k_out = k_flat.permute({1, 0, 2}).unsqueeze(0).contiguous();  // [1, H, total_len, D]
    auto v_out = v_flat.permute({1, 0, 2}).unsqueeze(0).contiguous();
    return {k_out, v_out};
  }

  BlockManager mgr_;
  int64_t n_layers_;
  int64_t page_size_;
  int64_t logical_len_ = 0;
  torch::Device device_;
};

}  // namespace

std::unique_ptr<IPagedKVCache> make_paged_kv_cache(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    int64_t max_pages,
    torch::Device device,
    torch::Dtype dtype) {
  return std::make_unique<PagedKVCache>(
      n_layers, n_kv_heads, head_dim, page_size, max_pages, device, dtype);
}

}  // namespace olmo_cpp
