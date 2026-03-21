#pragma once
#include <torch/torch.h>

namespace olmo_cpp {

/// Lion optimizer: uses sign of momentum for updates
/// Simpler than Adam, less memory (no second moment)
struct LionOptions : public torch::optim::OptimizerCloneableOptions<LionOptions> {
  LionOptions(double lr = 1e-4) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1e-4;
  TORCH_ARG(double, beta1) = 0.9;
  TORCH_ARG(double, beta2) = 0.99;
  TORCH_ARG(double, weight_decay) = 0.0;
  void set_lr(double lr) override { lr_ = lr; }
  double get_lr() const override { return lr_; }
};

class Lion : public torch::optim::Optimizer {
 public:
  explicit Lion(std::vector<torch::optim::OptimizerParamGroup> param_groups, LionOptions defaults = {});
  explicit Lion(std::vector<torch::Tensor> params, LionOptions defaults = {});
  using torch::optim::Optimizer::step;
  torch::Tensor step(LossClosure closure = nullptr) override;
};

}  // namespace olmo_cpp
