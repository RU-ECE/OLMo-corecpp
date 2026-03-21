#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <mutex>
#include <vector>

namespace olmo_cpp {

/// Arena-style scratch memory allocator for temporary tensors during forward/backward.
///
/// Eliminates repeated malloc/free overhead by pre-allocating large blocks and
/// sub-allocating from them. Works with begin_scope()/end_scope() on IBackend.
///
/// Usage:
///   Arena arena(64 * 1024 * 1024);  // 64 MB
///   {
///     ArenaScope scope(arena);
///     auto tmp1 = arena.allocate_tensor({batch, seq, dim}, torch::kFloat32);
///     auto tmp2 = arena.allocate_tensor({batch, seq, dim}, torch::kFloat32);
///     // ... use tmp1, tmp2 ...
///   }  // scope ends: all allocations freed at once (O(1) reset)
///
class Arena {
 public:
  /// Create arena with given capacity in bytes.
  explicit Arena(size_t capacity_bytes = 64 * 1024 * 1024);

  /// Allocate a tensor from the arena. Falls back to regular alloc if arena is full.
  /// The returned tensor is NOT initialized (contains garbage).
  torch::Tensor allocate_tensor(at::IntArrayRef sizes, torch::ScalarType dtype,
                                 torch::Device device = torch::kCPU);

  /// Reset the arena offset to the given mark (or 0). O(1) operation.
  void reset(size_t mark = 0);

  /// Get current allocation mark (for nested scopes).
  size_t mark() const { return offset_; }

  /// Total capacity in bytes.
  size_t capacity() const { return capacity_; }

  /// Bytes currently allocated.
  size_t used() const { return offset_; }

  /// Number of allocations that fell through to system malloc.
  int64_t fallback_count() const { return fallback_count_; }

 private:
  std::vector<uint8_t> buffer_;
  size_t capacity_;
  size_t offset_ = 0;
  int64_t fallback_count_ = 0;
};

/// RAII scope guard for arena. Saves the mark on construction, resets on destruction.
class ArenaScope {
 public:
  explicit ArenaScope(Arena& arena) : arena_(arena), saved_mark_(arena.mark()) {}
  ~ArenaScope() { arena_.reset(saved_mark_); }

  // Non-copyable, non-movable
  ArenaScope(const ArenaScope&) = delete;
  ArenaScope& operator=(const ArenaScope&) = delete;

 private:
  Arena& arena_;
  size_t saved_mark_;
};

/// Thread-local arena accessor. Each thread gets its own arena.
Arena& thread_arena();

/// Set the thread-local arena capacity (must be called before first use).
void set_arena_capacity(size_t bytes);

}  // namespace olmo_cpp
