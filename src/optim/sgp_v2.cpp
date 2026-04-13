#include "olmo_cpp/optim/sgp_v2.hpp"
#include <ATen/ATen.h>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace olmo_cpp {

SGPv2Predictor::SGPv2Predictor(const std::vector<torch::Tensor>& params,
                               SGPConfig config, int64_t rank)
    : config_(config), rank_(rank), params_(params),
      k_(config.initial_k), steps_since_anchor_(0) {
  states_.resize(params_.size());
  int64_t rank2d_count = 0;
  int64_t linear_count = 0;
  for (size_t i = 0; i < params_.size(); ++i) {
    auto& p = params_[i];
    auto& ps = states_[i];
    // A parameter qualifies for rank-r tracking if it is strictly 2D,
    // both dimensions are at least 2*rank, and the total element count
    // exceeds min_param_numel.
    if (p.dim() == 2
        && p.size(0) >= 2 * rank_
        && p.size(1) >= 2 * rank_
        && p.numel() >= config_.min_param_numel) {
      ps.mode = Mode::Rank2D;
      ps.m = p.size(0);
      ps.n = p.size(1);
      rank2d_count++;
    } else {
      ps.mode = Mode::Linear;
      if (p.numel() >= config_.min_param_numel) linear_count++;
    }
  }
  std::cout << "SGP v2: " << rank2d_count << " rank-" << rank_ << " params, "
            << linear_count << " linear-predictor params" << std::endl;
}

bool SGPv2Predictor::should_skip_backward(int64_t global_step) const {
  if (global_step < config_.warmup_steps) return false;
  if (steps_since_anchor_ == 0) return false;
  return steps_since_anchor_ < k_;
}

void SGPv2Predictor::update_basis(ParamState& ps, const torch::Tensor& G_2d) {
  // Randomized SVD (Halko et al.) for truncated rank-r approximation:
  //   Y = G Ω        (sketch the column space)
  //   Q = qr(Y)      (orthonormalize the sketch)
  //   B = Q^T G      (project G onto Q)
  //   svd(B) = U_b S V_h
  //   U_left  = Q U_b[:, :r]
  //   U_right = V_h[:r, :]^T
  int64_t oversample = 5;
  int64_t min_dim = std::min(ps.m, ps.n);
  int64_t r = std::min<int64_t>(rank_, min_dim - 1);
  if (r < 1) return;
  int64_t sketch = std::min<int64_t>(r + oversample, min_dim);

  auto Gf = G_2d.to(torch::kFloat);
  auto omega = torch::randn({ps.n, sketch}, Gf.options());
  auto Y = torch::matmul(Gf, omega);  // m × sketch
  auto qr_result = at::linalg_qr(Y, "reduced");
  auto Q = std::get<0>(qr_result);  // m × sketch
  auto B = torch::matmul(Q.transpose(0, 1), Gf);  // sketch × n

  auto svd_result = at::_linalg_svd(B, /*full_matrices=*/false, /*compute_uv=*/true);
  auto U_b = std::get<0>(svd_result);  // sketch × sketch (or sketch × min(sketch,n))
  auto Vh = std::get<2>(svd_result);   // min(sketch,n) × n

  int64_t r_eff = std::min<int64_t>(r, std::min(U_b.size(1), Vh.size(0)));
  if (r_eff < 1) return;
  auto U_left = torch::matmul(Q, U_b.slice(1, 0, r_eff));  // m × r
  auto U_right = Vh.slice(0, 0, r_eff).transpose(0, 1).contiguous();  // n × r

  ps.U_left = U_left.contiguous();
  ps.U_right = U_right;
  ps.has_basis = true;
}

void SGPv2Predictor::observe_real_gradients() {
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

    if (ps.mode == Mode::Linear) {
      // Same linear predictor as v1.
      if (ps.has_history && steps_since_anchor_ > 0) {
        auto predicted = ps.prev_grad * ps.alpha;
        if (ps.has_two_history) {
          predicted = predicted + ps.prev_prev_grad * ps.beta;
        }
        float residual = (true_grad - predicted).norm().item<float>();
        float true_norm = true_grad.norm().item<float>();
        if (true_norm > 1e-8f) {
          total_error += residual / true_norm;
          error_count++;
        }
        update_linear_coefficients(ps, true_grad);
      }
      if (ps.has_history) {
        ps.prev_prev_grad = ps.prev_grad;
        ps.has_two_history = true;
      }
      ps.prev_grad = true_grad.clone();
      ps.has_history = true;
      continue;
    }

    // Rank-r subspace path.
    auto G_2d = true_grad.view({ps.m, ps.n});

    // Prediction-quality measurement uses the *previous* basis (still valid
    // because we haven't updated yet), and the full decomposition:
    //   pred = U * extrap(coord_{t-1}, coord_{t-2}) * U^T  +  off_subspace_{t-1}
    if (ps.has_basis && ps.has_history && steps_since_anchor_ > 0) {
      auto Gf_prev = ps.prev_grad.to(torch::kFloat);
      auto prev_coord = torch::matmul(
          torch::matmul(ps.U_left.transpose(0, 1), Gf_prev),
          ps.U_right);  // r × r
      auto pred_coord = ps.alpha * prev_coord;
      if (ps.has_two_history) {
        auto Gf_pp = ps.prev_prev_grad.to(torch::kFloat);
        auto pp_coord = torch::matmul(
            torch::matmul(ps.U_left.transpose(0, 1), Gf_pp),
            ps.U_right);
        pred_coord = pred_coord + ps.beta * pp_coord;
      }
      auto rank_r_part = torch::matmul(
          torch::matmul(ps.U_left, pred_coord),
          ps.U_right.transpose(0, 1));
      auto off_subspace = Gf_prev - torch::matmul(
          torch::matmul(ps.U_left, prev_coord),
          ps.U_right.transpose(0, 1));
      auto predicted_f = rank_r_part + off_subspace;

      auto G_f = G_2d.to(torch::kFloat);
      float residual = (G_f - predicted_f).norm().item<float>();
      float true_norm = G_f.norm().item<float>();
      if (true_norm > 1e-8f) {
        total_error += residual / true_norm;
        error_count++;
      }
    }

    // Refresh basis from the current (real) gradient.
    update_basis(ps, G_2d);

    // Update linear coefficients in coordinate space, using the new basis.
    if (ps.has_basis && ps.has_history) {
      auto Gf_cur = G_2d.to(torch::kFloat);
      auto Gf_prev = ps.prev_grad.to(torch::kFloat);
      auto cur_coord = torch::matmul(
          torch::matmul(ps.U_left.transpose(0, 1), Gf_cur),
          ps.U_right);
      auto prev_coord = torch::matmul(
          torch::matmul(ps.U_left.transpose(0, 1), Gf_prev),
          ps.U_right);
      float dot = (cur_coord * prev_coord).sum().item<float>();
      float prev_norm_sq = (prev_coord * prev_coord).sum().item<float>();
      if (prev_norm_sq > 1e-12f) {
        float new_alpha = dot / prev_norm_sq;
        ps.alpha = 0.7f * ps.alpha + 0.3f * new_alpha;
      }
      if (ps.has_two_history) {
        auto Gf_pp = ps.prev_prev_grad.to(torch::kFloat);
        auto pp_coord = torch::matmul(
            torch::matmul(ps.U_left.transpose(0, 1), Gf_pp),
            ps.U_right);
        auto res = cur_coord - prev_coord * ps.alpha;
        float dot2 = (res * pp_coord).sum().item<float>();
        float pp_norm_sq = (pp_coord * pp_coord).sum().item<float>();
        if (pp_norm_sq > 1e-12f) {
          float new_beta = dot2 / pp_norm_sq;
          ps.beta = 0.7f * ps.beta + 0.3f * new_beta;
          ps.beta = std::max(-0.5f, std::min(0.5f, ps.beta));
        }
      }
    }

    // Shift history.
    if (ps.has_history) {
      ps.prev_prev_grad = ps.prev_grad;
      ps.has_two_history = true;
    }
    ps.prev_grad = G_2d.clone();
    ps.has_history = true;
  }

  if (error_count > 0) {
    last_error_ = total_error / error_count;
    if (last_error_ < config_.grow_threshold && k_ < config_.max_k) {
      k_++;
    } else if (last_error_ > config_.shrink_threshold && k_ > config_.min_k) {
      k_--;
    }
  }

  steps_since_anchor_ = 1;
}

void SGPv2Predictor::apply_predicted_gradients() {
  skipped_++;
  total_++;

  torch::NoGradGuard no_grad;

  for (size_t i = 0; i < params_.size(); ++i) {
    auto& p = params_[i];
    auto& ps = states_[i];
    if (!ps.has_history) continue;
    if (p.numel() < config_.min_param_numel) continue;

    torch::Tensor predicted;
    if (ps.mode == Mode::Linear) {
      predicted = ps.prev_grad * ps.alpha;
      if (ps.has_two_history) {
        predicted = predicted + ps.prev_prev_grad * ps.beta;
      }
    } else {
      if (!ps.has_basis) continue;
      auto Gf_prev = ps.prev_grad.to(torch::kFloat);
      auto prev_coord = torch::matmul(
          torch::matmul(ps.U_left.transpose(0, 1), Gf_prev),
          ps.U_right);
      auto pred_coord = ps.alpha * prev_coord;
      if (ps.has_two_history) {
        auto Gf_pp = ps.prev_prev_grad.to(torch::kFloat);
        auto pp_coord = torch::matmul(
            torch::matmul(ps.U_left.transpose(0, 1), Gf_pp),
            ps.U_right);
        pred_coord = pred_coord + ps.beta * pp_coord;
      }
      auto rank_r_part = torch::matmul(
          torch::matmul(ps.U_left, pred_coord),
          ps.U_right.transpose(0, 1));
      auto off_subspace = Gf_prev - torch::matmul(
          torch::matmul(ps.U_left, prev_coord),
          ps.U_right.transpose(0, 1));
      auto predicted_f = rank_r_part + off_subspace;
      predicted = predicted_f.to(ps.prev_grad.scalar_type()).view(p.sizes());
    }

    if (p.grad().defined()) {
      p.grad().copy_(predicted);
    } else {
      p.mutable_grad() = predicted.clone();
    }
  }

  steps_since_anchor_++;
}

void SGPv2Predictor::update_linear_coefficients(ParamState& ps, const torch::Tensor& true_grad) {
  if (!ps.has_history) return;
  auto g = true_grad.reshape(-1).to(torch::kFloat);
  auto g1 = ps.prev_grad.reshape(-1).to(torch::kFloat);
  float dot_g_g1 = (g * g1).sum().item<float>();
  float g1_norm_sq = (g1 * g1).sum().item<float>();
  if (g1_norm_sq > 1e-12f) {
    float new_alpha = dot_g_g1 / g1_norm_sq;
    ps.alpha = 0.7f * ps.alpha + 0.3f * new_alpha;
  }
  if (ps.has_two_history) {
    auto g2 = ps.prev_prev_grad.reshape(-1).to(torch::kFloat);
    auto residual = g - g1 * ps.alpha;
    float dot_r_g2 = (residual * g2).sum().item<float>();
    float g2_norm_sq = (g2 * g2).sum().item<float>();
    if (g2_norm_sq > 1e-12f) {
      float new_beta = dot_r_g2 / g2_norm_sq;
      ps.beta = 0.7f * ps.beta + 0.3f * new_beta;
      ps.beta = std::max(-0.5f, std::min(0.5f, ps.beta));
    }
  }
}

}  // namespace olmo_cpp
