#include "olmo_cpp/backend/arena.hpp"
#include <algorithm>
#include <cstring>

namespace olmo_cpp {

Arena::Arena(size_t capacity_bytes)
    : buffer_(capacity_bytes), capacity_(capacity_bytes) {}

torch::Tensor Arena::allocate_tensor(at::IntArrayRef sizes, torch::ScalarType dtype,
                                      torch::Device device) {
  // Only arena-allocate CPU tensors
  if (!device.is_cpu()) {
    fallback_count_++;
    return torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(device));
  }

  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  size_t elem_size = torch::elementSize(dtype);
  size_t bytes = static_cast<size_t>(numel) * elem_size;

  // Align to 64 bytes (cache line)
  size_t aligned_offset = (offset_ + 63) & ~size_t(63);
  size_t new_offset = aligned_offset + bytes;

  if (new_offset > capacity_) {
    // Arena full, fall back to system allocation
    fallback_count_++;
    return torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(device));
  }

  // Create tensor that points into arena buffer (no-op deleter: arena owns memory)
  void* ptr = buffer_.data() + aligned_offset;
  offset_ = new_offset;

  // Use from_blob with a custom deleter that does nothing (arena manages lifetime)
  auto tensor = torch::from_blob(
      ptr, sizes,
      /*deleter=*/[](void*) {},  // no-op: arena owns the memory
      torch::TensorOptions().dtype(dtype).device(torch::kCPU));

  return tensor;
}

void Arena::reset(size_t mark) {
  offset_ = std::min(mark, capacity_);
}

// ---------------------------------------------------------------------------
// Thread-local arena
// ---------------------------------------------------------------------------

static thread_local size_t tl_arena_capacity = 64 * 1024 * 1024;  // 64 MB default
static thread_local std::unique_ptr<Arena> tl_arena;

Arena& thread_arena() {
  if (!tl_arena) {
    tl_arena = std::make_unique<Arena>(tl_arena_capacity);
  }
  return *tl_arena;
}

void set_arena_capacity(size_t bytes) {
  tl_arena_capacity = bytes;
  tl_arena.reset();  // force re-creation on next access
}

}  // namespace olmo_cpp
