#pragma once

#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include <functional>

namespace olmo_cpp {

/// Mixed precision (BF16) training with autocast.
/// Uses at::autocast::set_autocast_enabled for CPU; CUDA uses device-specific API.
class AMPContext {
 public:
  explicit AMPContext(bool enabled = true) : enabled_(enabled) {}

  /// Run fn under autocast if enabled. Enables autocast, runs fn, restores.
  template <typename F>
  auto run(F&& fn) -> decltype(fn()) {
    if (!enabled_) return fn();
    auto device = at::DeviceType::CPU;  // Use CUDA when available
    bool prev = at::autocast::is_autocast_enabled(device);
    at::autocast::set_autocast_enabled(device, true);
    auto result = fn();
    at::autocast::set_autocast_enabled(device, prev);
    return result;
  }

  bool enabled() const { return enabled_; }

 private:
  bool enabled_;
};

/// Activation checkpointing: recompute forward in backward to save memory.
/// Placeholder - when enabled, just runs fn (full implementation needs custom autograd).
inline torch::Tensor checkpoint(
    const std::function<torch::Tensor()>& fn,
    const std::vector<torch::Tensor>& /*inputs*/) {
  return fn();
}

}  // namespace olmo_cpp
