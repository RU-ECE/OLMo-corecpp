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

  // Reuse a thread-local scratch vector across calls to avoid a per-step
  // heap allocation for the grad handle list.
  static thread_local std::vector<torch::Tensor> grads;
  grads.clear();
  size_t total_params = 0;
  for (auto& group : optimizer.param_groups()) total_params += group.params().size();
  grads.reserve(total_params);
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

  // Per-grad isfinite().all() produces a 0-dim bool per grad; stack them
  // and reduce once on-device. The previous implementation materialized a
  // single flat tensor of every gradient (O(model_bytes) alloc + copy!) just
  // to run one isfinite — unacceptable even on cold paths.
  static thread_local std::vector<torch::Tensor> finite_flags;
  finite_flags.clear();
  finite_flags.reserve(grads.size());
  for (const auto& g : grads) {
    finite_flags.push_back(torch::isfinite(g).all());
  }
  auto all_finite = torch::stack(finite_flags).all();

  if (!all_finite.item<bool>()) {
    found_inf_ = true;
    // Zero out all grads in one fused kernel
    at::_foreach_mul_(grads, 0.0);
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
