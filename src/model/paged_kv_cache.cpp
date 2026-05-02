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

#include <stdexcept>

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

}  // namespace olmo_cpp
