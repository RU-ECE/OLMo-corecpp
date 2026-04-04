#include "olmo_cpp/train/grad_scaler.hpp"
#include <ATen/ops/_foreach_mul.h>
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

  // Collect all defined gradients
  std::vector<torch::Tensor> grads;
  for (auto& group : optimizer.param_groups()) {
    for (auto& p : group.params()) {
      if (p.grad().defined()) {
        grads.push_back(p.grad());
      }
    }
  }

  if (grads.empty()) return true;

  // Batched unscale: 1 fused kernel launch instead of N individual mul_ calls
  at::_foreach_mul_(grads, static_cast<double>(inv_scale));

  // Batched finite check: cat all grads into a single flat tensor,
  // then one isfinite().all() check (still needs 1 D2H sync for the bool)
  std::vector<torch::Tensor> flat_grads;
  flat_grads.reserve(grads.size());
  for (auto& g : grads) {
    flat_grads.push_back(g.reshape(-1));
  }
  auto all_grads = torch::cat(flat_grads);
  auto all_finite = torch::isfinite(all_grads).all();

  if (!all_finite.item<bool>()) {
    found_inf_ = true;
    // Zero out all grads in one pass
    for (auto& g : grads) {
      g.zero_();
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
