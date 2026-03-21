#include "olmo_cpp/optim/lion.hpp"

namespace olmo_cpp {

namespace {

/// Per-parameter state for Lion optimizer
struct LionParamState : public torch::optim::OptimizerCloneableParamState<LionParamState> {
  TORCH_ARG(torch::Tensor, momentum_buffer);
  TORCH_ARG(int64_t, step) = 0;

  void serialize(torch::serialize::OutputArchive& archive) const override {
    archive.write("step", torch::scalar_tensor(step(), torch::kInt64));
    if (momentum_buffer().defined()) {
      archive.write("momentum_buffer", momentum_buffer());
    }
  }

  void serialize(torch::serialize::InputArchive& archive) override {
    torch::Tensor t;
    archive.read("step", t);
    step(t.item<int64_t>());
    torch::Tensor buf;
    if (archive.try_read("momentum_buffer", buf)) {
      momentum_buffer(buf);
    }
  }
};

}  // namespace

Lion::Lion(std::vector<torch::Tensor> params, LionOptions defaults)
    : Optimizer(
          {torch::optim::OptimizerParamGroup(std::move(params))},
          std::make_unique<LionOptions>(defaults)) {}

Lion::Lion(std::vector<torch::optim::OptimizerParamGroup> param_groups, LionOptions defaults)
    : Optimizer(
          std::move(param_groups),
          std::make_unique<LionOptions>(defaults)) {}

torch::Tensor Lion::step(LossClosure closure) {
  torch::NoGradGuard no_grad;
  torch::Tensor loss = {};
  if (closure) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  for (auto& group : param_groups_) {
    auto& options = static_cast<LionOptions&>(group.options());
    const double lr = options.lr();
    const double beta1 = options.beta1();
    const double beta2 = options.beta2();
    const double weight_decay = options.weight_decay();

    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }

      auto grad = p.grad();
      auto key = p.unsafeGetTensorImpl();

      // Initialize state if needed
      if (state_.find(key) == state_.end()) {
        auto s = std::make_unique<LionParamState>();
        s->momentum_buffer(torch::zeros_like(p.data()));
        s->step(0);
        state_[key] = std::move(s);
      }

      auto& state = static_cast<LionParamState&>(*state_[key]);
      auto& m = state.momentum_buffer();
      state.step(state.step() + 1);

      // Step 1: Weight decay (decoupled)
      if (weight_decay != 0.0) {
        p.data().add_(p.data(), -lr * weight_decay);
      }

      // Step 2: Compute update direction = sign(beta1 * m + (1 - beta1) * grad)
      // Use a temporary to avoid modifying momentum prematurely
      auto update = (m * beta1 + grad * (1.0 - beta1)).sign_();

      // Step 3: Apply update
      p.data().add_(update, -lr);

      // Step 4: Update momentum for next step: m = beta2 * m + (1 - beta2) * g
      m.mul_(beta2).add_(grad, 1.0 - beta2);
    }
  }

  return loss;
}

}  // namespace olmo_cpp
