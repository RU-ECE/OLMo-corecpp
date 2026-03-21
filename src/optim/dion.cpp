#include "olmo_cpp/optim/dion.hpp"

namespace olmo_cpp {

namespace {

/// Per-parameter state for DION optimizer
struct DIONParamState : public torch::optim::OptimizerCloneableParamState<DIONParamState> {
  TORCH_ARG(torch::Tensor, exp_avg);      // First moment (m)
  TORCH_ARG(torch::Tensor, exp_avg_sq);   // Second moment (v) - diagonal Hessian approx
  TORCH_ARG(int64_t, step) = 0;

  void serialize(torch::serialize::OutputArchive& archive) const override {
    archive.write("step", torch::scalar_tensor(step(), torch::kInt64));
    if (exp_avg().defined()) {
      archive.write("exp_avg", exp_avg());
    }
    if (exp_avg_sq().defined()) {
      archive.write("exp_avg_sq", exp_avg_sq());
    }
  }

  void serialize(torch::serialize::InputArchive& archive) override {
    torch::Tensor t;
    archive.read("step", t);
    step(t.item<int64_t>());
    torch::Tensor buf;
    if (archive.try_read("exp_avg", buf)) {
      exp_avg(buf);
    }
    if (archive.try_read("exp_avg_sq", buf)) {
      exp_avg_sq(buf);
    }
  }
};

}  // namespace

DION::DION(std::vector<torch::Tensor> params, DIONOptions defaults)
    : Optimizer(
          {torch::optim::OptimizerParamGroup(std::move(params))},
          std::make_unique<DIONOptions>(defaults)) {}

DION::DION(std::vector<torch::optim::OptimizerParamGroup> param_groups, DIONOptions defaults)
    : Optimizer(
          std::move(param_groups),
          std::make_unique<DIONOptions>(defaults)) {}

torch::Tensor DION::step(LossClosure closure) {
  torch::NoGradGuard no_grad;
  torch::Tensor loss = {};
  if (closure) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  for (auto& group : param_groups_) {
    auto& options = static_cast<DIONOptions&>(group.options());
    const double lr = options.lr();
    const double beta1 = options.beta1();
    const double beta2 = options.beta2();
    const double eps = options.eps();
    const double weight_decay = options.weight_decay();

    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }

      auto grad = p.grad();
      auto key = p.unsafeGetTensorImpl();

      // Initialize state if needed
      if (state_.find(key) == state_.end()) {
        auto s = std::make_unique<DIONParamState>();
        s->exp_avg(torch::zeros_like(p.data()));
        s->exp_avg_sq(torch::zeros_like(p.data()));
        s->step(0);
        state_[key] = std::move(s);
      }

      auto& state = static_cast<DIONParamState&>(*state_[key]);
      auto& m = state.exp_avg();
      auto& v = state.exp_avg_sq();
      state.step(state.step() + 1);
      int64_t t = state.step();

      // Step 1: Update biased first moment estimate
      // m = beta1 * m + (1 - beta1) * g
      m.mul_(beta1).add_(grad, 1.0 - beta1);

      // Step 2: Update biased second moment estimate (diagonal Hessian approximation)
      // v = beta2 * v + (1 - beta2) * g^2
      v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

      // Step 3: Bias correction
      double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(t));
      double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(t));
      auto m_hat = m / bias_correction1;
      auto v_hat = v / bias_correction2;

      // Step 4: Parameter update
      // p -= lr * m_hat / (sqrt(v_hat) + eps)
      p.data().addcdiv_(m_hat, v_hat.sqrt().add_(eps), -lr);

      // Step 5: Weight decay (decoupled, applied after adaptive update)
      if (weight_decay != 0.0) {
        p.data().add_(p.data(), -lr * weight_decay);
      }
    }
  }

  return loss;
}

}  // namespace olmo_cpp
