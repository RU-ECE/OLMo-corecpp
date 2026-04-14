#pragma once

#include "zwt/core/device.hpp"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace zwt {

// Minimal allocator interface. Two concrete impls below:
//  * ArenaAllocator — bump-pointer, per-stream scratch; reset() frees all at once.
//  * PoolAllocator  — size-bucketed freelist for long-lived tensors
//                     (parameters, optimizer state, persistent buffers).
//
// Neither uses refcounts. Tensor owns its storage explicitly; on destruction
// the allocator is called to release. No PyTorch-style caching indirection.
class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual void* alloc(size_t bytes, size_t alignment = 256) = 0;
  virtual void  free(void* ptr, size_t bytes) = 0;
  virtual Device device() const = 0;
};

// Bump-pointer arena. Perfect for activations: allocate during forward,
// reset() at end of step. Zero per-alloc bookkeeping, zero fragmentation.
class ArenaAllocator final : public Allocator {
 public:
  ArenaAllocator(Device dev, size_t capacity_bytes);
  ~ArenaAllocator() override;

  void* alloc(size_t bytes, size_t alignment = 256) override;
  void  free(void* ptr, size_t bytes) override;  // no-op; reset drops everything
  Device device() const override { return device_; }

  void   reset()      { offset_ = 0; }
  size_t mark() const { return offset_; }
  void   rewind(size_t m) { offset_ = m; }
  size_t used() const { return offset_; }
  size_t capacity() const { return capacity_; }

 private:
  Device device_;
  void*  base_     = nullptr;
  size_t capacity_ = 0;
  size_t offset_   = 0;
};

// Size-bucketed pool. Parameters, gradients, optimizer state live here —
// they persist across steps but are freed exactly once. We don't need a
// general-purpose allocator; we need the three or four sizes training uses.
class PoolAllocator final : public Allocator {
 public:
  explicit PoolAllocator(Device dev);
  ~PoolAllocator() override;

  void* alloc(size_t bytes, size_t alignment = 256) override;
  void  free(void* ptr, size_t bytes) override;
  Device device() const override { return device_; }

  size_t live_bytes()  const { return live_bytes_; }
  size_t cached_bytes() const { return cached_bytes_; }

 private:
  struct Block { void* ptr; size_t size; };

  void* raw_alloc(size_t bytes);
  void  raw_free(void* p);

  Device device_;
  // One freelist per power-of-two bucket. Index = ceil(log2(size)).
  std::vector<std::vector<Block>> buckets_;
  std::mutex mu_;
  size_t live_bytes_ = 0;
  size_t cached_bytes_ = 0;
};

// Process-wide allocator lookup. You normally touch these two:
//   device_pool(Device)    — long-lived params/grads/optimizer state
//   activation_arena(Device) — per-step scratch; must be reset each step
Allocator& device_pool(Device dev);
Allocator& activation_arena(Device dev);

// Tuning hook: set the activation arena capacity before the first use.
// Default is 1 GiB on CUDA, 256 MiB on CPU.
void set_activation_arena_capacity(size_t bytes);

}  // namespace zwt
