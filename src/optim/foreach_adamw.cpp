#include "olmo_cpp/optim/foreach_adamw.hpp"
#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_addcmul.h>
#include <ATen/ops/_foreach_addcdiv.h>
#include <ATen/ops/_foreach_sqrt.h>
#include <cmath>

namespace olmo_cpp {

namespace {

/// Per-parameter state for ForeachAdamW
struct ForeachAdamWParamState
    : public torch::optim::OptimizerCloneableParamState<ForeachAdamWParamState> {
  TORCH_ARG(torch::Tensor, exp_avg);
  TORCH_ARG(torch::Tensor, exp_avg_sq);

  void serialize(torch::serialize::OutputArchive& archive) const override {
    if (exp_avg().defined()) archive.write("exp_avg", exp_avg());
    if (exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq());
  }

  void serialize(torch::serialize::InputArchive& archive) override {
    torch::Tensor t;
    if (archive.try_read("exp_avg", t)) exp_avg(t);
    if (archive.try_read("exp_avg_sq", t)) exp_avg_sq(t);
  }
};

}  // namespace

ForeachAdamW::ForeachAdamW(std::vector<torch::Tensor> params, ForeachAdamWOptions defaults)
    : Optimizer(
          {torch::optim::OptimizerParamGroup(std::move(params))},
          std::make_unique<ForeachAdamWOptions>(defaults)) {}

ForeachAdamW::ForeachAdamW(std::vector<torch::optim::OptimizerParamGroup> param_groups,
                           ForeachAdamWOptions defaults)
    : Optimizer(
          std::move(param_groups),
          std::make_unique<ForeachAdamWOptions>(defaults)) {}

torch::Tensor ForeachAdamW::step(LossClosure closure) {
  torch::NoGradGuard no_grad;
  torch::Tensor loss = {};
  if (closure) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  step_count_++;

  // Collect all params, grads, and state tensors into parallel vectors
  // across all param groups (typically just one group for AdamW)
  for (auto& group : param_groups_) {
    auto& options = static_cast<ForeachAdamWOptions&>(group.options());
    const double lr = options.lr();
    const double beta1 = options.beta1();
    const double beta2 = options.beta2();
    const double eps = options.eps();
    const double weight_decay = options.weight_decay();

    // Reuse the scratch vectors across steps — clear() retains capacity so
    // we pay for one allocation per vector on the first step, zero on every
    // step thereafter.
    auto& params_vec = params_scratch_;
    auto& grads_vec = grads_scratch_;
    auto& exp_avg_vec = exp_avg_scratch_;
    auto& exp_avg_sq_vec = exp_avg_sq_scratch_;
    params_vec.clear();
    grads_vec.clear();
    exp_avg_vec.clear();
    exp_avg_sq_vec.clear();
    const size_t n_params = group.params().size();
    params_vec.reserve(n_params);
    grads_vec.reserve(n_params);
    exp_avg_vec.reserve(n_params);
    exp_avg_sq_vec.reserve(n_params);

    for (auto& p : group.params()) {
      if (!p.grad().defined()) continue;

      auto key = p.unsafeGetTensorImpl();

      // Lazy-init state on first step
      if (state_.find(key) == state_.end()) {
        auto s = std::make_unique<ForeachAdamWParamState>();
        s->exp_avg(torch::zeros_like(p.data()));
        s->exp_avg_sq(torch::zeros_like(p.data()));
        state_[key] = std::move(s);
      }

      auto& state = static_cast<ForeachAdamWParamState&>(*state_[key]);
      params_vec.push_back(p.data());
      grads_vec.push_back(p.grad());
      exp_avg_vec.push_back(state.exp_avg());
      exp_avg_sq_vec.push_back(state.exp_avg_sq());
    }

    if (params_vec.empty()) continue;

    // Bias correction factors
    double bc1 = 1.0 - std::pow(beta1, step_count_);
    double bc2 = 1.0 - std::pow(beta2, step_count_);
    double bc2_sqrt = std::sqrt(bc2);
    // Fold bc2_sqrt into step_size and eps to eliminate one kernel launch:
    //   p -= (lr/bc1) * m / (sqrt(v)/bc2_sqrt + eps)
    // = p -= (lr*bc2_sqrt/bc1) * m / (sqrt(v) + eps*bc2_sqrt)
    double step_size = -lr * bc2_sqrt / bc1;
    double eps_scaled = eps * bc2_sqrt;

    // === Fused operations: 7 kernel launches (6 without weight decay) ===

    // 1. Decoupled weight decay: p *= (1 - lr * weight_decay)
    if (weight_decay != 0.0) {
      at::_foreach_mul_(params_vec, 1.0 - lr * weight_decay);
    }

    // 2-3. Update biased first moment: m = beta1 * m + (1 - beta1) * g
    at::_foreach_mul_(exp_avg_vec, beta1);
    at::_foreach_add_(exp_avg_vec, grads_vec, 1.0 - beta1);

    // 4-5. Update biased second moment: v = beta2 * v + (1 - beta2) * g^2
    at::_foreach_mul_(exp_avg_sq_vec, beta2);
    at::_foreach_addcmul_(exp_avg_sq_vec, grads_vec, grads_vec, 1.0 - beta2);

    // 6. Compute denominator: denom = sqrt(v) + eps * bc2_sqrt
    //    (bc2_sqrt is folded into step_size, eliminating the division kernel)
    auto denom = at::_foreach_sqrt(exp_avg_sq_vec);
    at::_foreach_add_(denom, eps_scaled);

    // 7. Apply update: p += step_size * m / denom
    at::_foreach_addcdiv_(params_vec, exp_avg_vec, denom, step_size);
  }

  return loss;
}

}  // namespace olmo_cpp
