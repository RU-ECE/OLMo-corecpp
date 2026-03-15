#pragma once

#include <torch/torch.h>

namespace olmo_cpp {

/// Gradient scaler for mixed precision training.
/// Scales loss to prevent float16 gradient underflow.
class GradScaler {
 public:
  GradScaler(float init_scale = 65536.0f, float growth_factor = 2.0f,
             float backoff_factor = 0.5f, int growth_interval = 2000);

  /// Scale loss before backward()
  torch::Tensor scale(torch::Tensor loss);

  /// Unscale gradients. Returns true if all grads are finite.
  bool unscale_and_check(torch::optim::Optimizer& optimizer);

  /// Step optimizer (only if grads are finite)
  void step(torch::optim::Optimizer& optimizer);

  /// Update scale factor after step
  void update();

  float current_scale() const { return scale_; }
  bool found_inf() const { return found_inf_; }

 private:
  float scale_;
  float growth_factor_;
  float backoff_factor_;
  int growth_interval_;
  int steps_since_growth_ = 0;
  bool found_inf_ = false;
};

}  // namespace olmo_cpp
