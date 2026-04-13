#include "olmo_cpp/optim/sgp.hpp"
#include <iostream>
#include <cmath>

namespace olmo_cpp {

SGPPredictor::SGPPredictor(const std::vector<torch::Tensor>& params, SGPConfig config)
    : config_(config), params_(params), k_(config.initial_k), steps_since_anchor_(0) {
  states_.resize(params_.size());
}

bool SGPPredictor::should_skip_backward(int64_t global_step) const {
  if (global_step < config_.warmup_steps) return false;
  if (steps_since_anchor_ == 0) return false;  // just anchored, next one is real
  return steps_since_anchor_ < k_;
}

void SGPPredictor::observe_real_gradients() {
  total_++;
  double total_error = 0.0;
  int64_t error_count = 0;

  torch::NoGradGuard no_grad;

  for (size_t i = 0; i < params_.size(); ++i) {
    auto& p = params_[i];
    auto& ps = states_[i];
    if (!p.grad().defined()) continue;
    if (p.numel() < config_.min_param_numel) continue;

    auto true_grad = p.grad().detach();

    // Measure prediction quality if we have history
    if (ps.has_history && steps_since_anchor_ > 0) {
      // Compute what we would have predicted
      auto predicted = ps.prev_grad * ps.alpha;
      if (ps.has_two_history) {
        predicted = predicted + ps.prev_prev_grad * ps.beta;
      }

      auto residual = (true_grad - predicted).norm().item<float>();
      auto true_norm = true_grad.norm().item<float>();
      if (true_norm > 1e-8f) {
        total_error += residual / true_norm;
        error_count++;
      }

      update_predictor_coefficients(ps, true_grad);
    }

    // Shift history
    if (ps.has_history) {
      ps.prev_prev_grad = ps.prev_grad;
      ps.has_two_history = true;
    }
    ps.prev_grad = true_grad.clone();
    ps.has_history = true;
  }

  // Adapt K based on prediction error
  if (error_count > 0) {
    last_error_ = total_error / error_count;

    if (last_error_ < config_.grow_threshold && k_ < config_.max_k) {
      k_++;
    } else if (last_error_ > config_.shrink_threshold && k_ > config_.min_k) {
      k_--;
    }
  }

  steps_since_anchor_ = 1;  // reset: next K-1 steps can be predicted
}

void SGPPredictor::apply_predicted_gradients() {
  skipped_++;
  total_++;

  torch::NoGradGuard no_grad;

  for (size_t i = 0; i < params_.size(); ++i) {
    auto& p = params_[i];
    auto& ps = states_[i];
    if (!ps.has_history) continue;
    if (p.numel() < config_.min_param_numel) continue;

    // Linear predictor: G_pred = alpha * G_{t-1} + beta * G_{t-2}
    auto predicted = ps.prev_grad * ps.alpha;
    if (ps.has_two_history) {
      predicted = predicted + ps.prev_prev_grad * ps.beta;
    }

    // Set gradient directly
    if (p.grad().defined()) {
      p.grad().copy_(predicted);
    } else {
      p.mutable_grad() = predicted.clone();
    }
  }

  steps_since_anchor_++;
}

void SGPPredictor::update_predictor_coefficients(ParamState& ps, const torch::Tensor& true_grad) {
  // Online least-squares update of alpha, beta using exponential moving average.
  // We minimize ||G_true - alpha * G_{t-1} - beta * G_{t-2}||^2
  //
  // Simplified: use the cosine similarity between G_true and G_{t-1} as alpha,
  // and the residual's alignment with G_{t-2} as beta. This avoids storing
  // covariance matrices.
  if (!ps.has_history) return;

  auto g = true_grad.reshape(-1).to(torch::kFloat);
  auto g1 = ps.prev_grad.reshape(-1).to(torch::kFloat);

  float dot_g_g1 = (g * g1).sum().item<float>();
  float g1_norm_sq = (g1 * g1).sum().item<float>();

  if (g1_norm_sq > 1e-12f) {
    // Optimal alpha (projection coefficient): <G, G_{t-1}> / ||G_{t-1}||^2
    float new_alpha = dot_g_g1 / g1_norm_sq;
    // EMA smoothing to avoid oscillation
    ps.alpha = 0.7f * ps.alpha + 0.3f * new_alpha;
  }

  if (ps.has_two_history) {
    auto g2 = ps.prev_prev_grad.reshape(-1).to(torch::kFloat);
    // Residual after alpha * G_{t-1}
    auto residual = g - g1 * ps.alpha;
    float dot_r_g2 = (residual * g2).sum().item<float>();
    float g2_norm_sq = (g2 * g2).sum().item<float>();

    if (g2_norm_sq > 1e-12f) {
      float new_beta = dot_r_g2 / g2_norm_sq;
      ps.beta = 0.7f * ps.beta + 0.3f * new_beta;
      // Clamp beta to prevent runaway extrapolation
      ps.beta = std::max(-0.5f, std::min(0.5f, ps.beta));
    }
  }
}

}  // namespace olmo_cpp
