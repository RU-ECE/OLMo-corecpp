#include "olmo_cpp/profiler.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace olmo_cpp {

Profiler& profiler() {
  static Profiler instance;
  return instance;
}

MemoryStats get_memory_stats(torch::Device device) {
  MemoryStats stats;

#ifdef USE_CUDA
  if (device.is_cuda() && torch::cuda::is_available()) {
    // Use CUDA memory info API
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    stats.reserved_bytes = static_cast<int64_t>(total_bytes);
    stats.allocated_bytes = static_cast<int64_t>(total_bytes - free_bytes);
    stats.peak_allocated_bytes = stats.allocated_bytes;  // approximation
  }
#endif
  // MPS doesn't expose detailed memory stats through LibTorch C++ API
  (void)device;
  return stats;
}

void print_memory_summary(torch::Device device) {
  auto stats = get_memory_stats(device);

  auto mb = [](int64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); };

  std::cout << "\n[Memory] Device: " << device << "\n";

  if (stats.allocated_bytes > 0 || stats.reserved_bytes > 0) {
    std::cout << "  Allocated: " << std::fixed << std::setprecision(1) << mb(stats.allocated_bytes) << " MB\n"
              << "  Reserved:  " << std::fixed << std::setprecision(1) << mb(stats.reserved_bytes) << " MB\n"
              << "  Peak:      " << std::fixed << std::setprecision(1) << mb(stats.peak_allocated_bytes) << " MB\n";
  } else {
    std::cout << "  (Detailed GPU memory stats not available for this device.)\n"
              << "  Use --profile with CUDA for detailed memory tracking.\n";
  }
  std::cout << std::endl;
}

}  // namespace olmo_cpp
