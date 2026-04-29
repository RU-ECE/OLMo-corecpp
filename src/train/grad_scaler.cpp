/**
 * src/train/grad_scaler.cpp
 *
 * ─── What "gradient scaling" is ─────────────────────────────────────
 *
 * When training in FP16 (not BF16), gradients can underflow. FP16
 * has only ~5 mantissa bits and a much narrower dynamic range than
 * FP32, so a small but non-zero gradient like 1e-8 quietly becomes
 * 0 — that gradient is then lost to the optimizer.
 *
 * The fix: multiply the loss by a large **scale factor** S before
 * calling backward. By the chain rule, every gradient is multiplied
 * by S too, lifting tiny values back into FP16's representable
 * range. Right before the optimizer step, divide the gradients by
 * S to recover the true values.
 *
 *   loss_scaled = loss * S
 *   loss_scaled.backward()         // grads now in FP16-friendly range
 *   if any grad is NaN/Inf:
 *     skip the step, halve S (`backoff_factor`)
 *   else:
 *     unscale grads by 1/S
 *     optim.step()
 *     every `growth_interval` clean steps, multiply S by `growth_factor`
 *
 * S is **dynamic**: it auto-adjusts up when training is calm and down
 * when an Inf appears, so the user doesn't need to tune it.
 *
 * BF16 doesn't underflow as easily, so this is mostly a no-op in the
 * BF16 path (use_bf16=1 in the .conf bypasses the scaler).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/train/grad_scaler.hpp : GradScaler declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: when train_cfg.use_grad_scaler=1, the train loop
 *     wraps loss.backward / optim.step with scale/unscale calls.
 *
 * --- Role in training pipeline ---
 *   FP16 precision plumbing. Off by default in the quickstart flow.
 */
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
