#pragma once
#include <torch/torch.h>

namespace olmo_cpp {

/// DION: Diagonal adaptive learning rate optimizer
struct DIONOptions : public torch::optim::OptimizerCloneableOptions<DIONOptions> {
  DIONOptions(double lr = 1e-3) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1e-3;
  TORCH_ARG(double, beta1) = 0.9;
  TORCH_ARG(double, beta2) = 0.999;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0.0;
  void set_lr(double lr) override { lr_ = lr; }
  double get_lr() const override { return lr_; }
};

class DION : public torch::optim::Optimizer {
 public:
  explicit DION(std::vector<torch::optim::OptimizerParamGroup> param_groups, DIONOptions defaults = {});
  explicit DION(std::vector<torch::Tensor> params, DIONOptions defaults = {});
  using torch::optim::Optimizer::step;
  torch::Tensor step(LossClosure closure = nullptr) override;
};

}  // namespace olmo_cpp
