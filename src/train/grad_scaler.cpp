#include "olmo_cpp/train/grad_scaler.hpp"
#include <cmath>

namespace olmo_cpp {

GradScaler::GradScaler(float init_scale, float growth_factor,
                       float backoff_factor, int growth_interval)
    : scale_(init_scale),
      growth_factor_(growth_factor),
      backoff_factor_(backoff_factor),
      growth_interval_(growth_interval) {}

torch::Tensor GradScaler::scale(torch::Tensor loss) {
  return loss * scale_;
}

bool GradScaler::unscale_and_check(torch::optim::Optimizer& optimizer) {
  found_inf_ = false;
  float inv_scale = 1.0f / scale_;

  // Unscale all gradients first
  std::vector<torch::Tensor> grads;
  for (auto& group : optimizer.param_groups()) {
    for (auto& p : group.params()) {
      if (p.grad().defined()) {
        p.grad().mul_(inv_scale);
        grads.push_back(p.grad());
      }
    }
  }

  // Single batched inf/nan check — one CUDA sync instead of per-parameter
  if (!grads.empty()) {
    auto all_finite = torch::ones({1}, grads[0].options());
    for (auto& g : grads) {
      all_finite.mul_(torch::isfinite(g).all().to(all_finite.dtype()));
    }
    if (!all_finite.item<bool>()) {
      found_inf_ = true;
      for (auto& g : grads) {
        g.zero_();
      }
    }
  }

  return !found_inf_;
}

void GradScaler::step(torch::optim::Optimizer& optimizer) {
  if (!found_inf_) {
    optimizer.step();
  }
}

void GradScaler::update() {
  if (found_inf_) {
    scale_ *= backoff_factor_;
    steps_since_growth_ = 0;
  } else {
    steps_since_growth_++;
    if (steps_since_growth_ >= growth_interval_) {
      scale_ *= growth_factor_;
      steps_since_growth_ = 0;
    }
  }
}

}  // namespace olmo_cpp
