#pragma once

#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#ifndef __APPLE__
#include <c10/core/impl/LocalDispatchKeySet.h>
#endif
#include <optional>

namespace olmo_cpp {

/// RAII guard that enables BF16 autocast on CUDA.
/// Portable across PyTorch 2.0 through 2.6+.
struct AutocastGuard {
  explicit AutocastGuard(bool enabled, torch::Device device)
      : enabled_(enabled && device.is_cuda()) {
    if (enabled_) {
#if defined(OLMO_AUTOCAST_DEVICE_API)
      prev_enabled_ = at::autocast::is_autocast_enabled(at::kCUDA);
      prev_dtype_ = at::autocast::get_autocast_dtype(at::kCUDA);
      at::autocast::set_autocast_enabled(at::kCUDA, true);
      at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
      at::autocast::increment_nesting();
#elif defined(OLMO_AUTOCAST_GPU_API)
      prev_enabled_ = at::autocast::is_autocast_gpu_enabled();
      prev_dtype_ = at::autocast::get_autocast_gpu_dtype();
      at::autocast::set_autocast_gpu_enabled(true);
      at::autocast::set_autocast_gpu_dtype(at::kBFloat16);
      at::autocast::increment_nesting();
#else
      dk_guard_.emplace(c10::DispatchKey::AutocastCUDA);
#endif
    }
  }

  ~AutocastGuard() {
    if (enabled_) {
#if defined(OLMO_AUTOCAST_DEVICE_API)
      at::autocast::decrement_nesting();
      at::autocast::clear_cache();
      at::autocast::set_autocast_enabled(at::kCUDA, prev_enabled_);
      at::autocast::set_autocast_dtype(at::kCUDA, prev_dtype_);
#elif defined(OLMO_AUTOCAST_GPU_API)
      at::autocast::decrement_nesting();
      at::autocast::clear_cache();
      at::autocast::set_autocast_gpu_enabled(prev_enabled_);
      at::autocast::set_autocast_gpu_dtype(prev_dtype_);
#else
      // dk_guard_ destructor removes the dispatch key automatically
#endif
    }
  }

  AutocastGuard(const AutocastGuard&) = delete;
  AutocastGuard& operator=(const AutocastGuard&) = delete;

 private:
  bool enabled_;
#if defined(OLMO_AUTOCAST_DEVICE_API) || defined(OLMO_AUTOCAST_GPU_API)
  bool prev_enabled_{false};
  at::ScalarType prev_dtype_{at::kFloat};
#else
  std::optional<c10::impl::IncludeDispatchKeyGuard> dk_guard_;
#endif
};

}  // namespace olmo_cpp
