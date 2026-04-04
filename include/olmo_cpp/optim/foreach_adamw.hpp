#pragma once
#include <torch/torch.h>

namespace olmo_cpp {

/// ForeachAdamW options — mirrors torch::optim::AdamWOptions
struct ForeachAdamWOptions : public torch::optim::OptimizerCloneableOptions<ForeachAdamWOptions> {
  ForeachAdamWOptions(double lr = 1e-3) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1e-3;
  TORCH_ARG(double, beta1) = 0.9;
  TORCH_ARG(double, beta2) = 0.999;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0.01;
  void set_lr(double lr) override { lr_ = lr; }
  double get_lr() const override { return lr_; }
};

/// Fused AdamW using _foreach_* batched operations.
/// ~7 kernel launches per step regardless of parameter count,
/// vs ~53,000 with LibTorch's default per-parameter AdamW.
class ForeachAdamW : public torch::optim::Optimizer {
 public:
  explicit ForeachAdamW(std::vector<torch::Tensor> params, ForeachAdamWOptions defaults = {});
  explicit ForeachAdamW(std::vector<torch::optim::OptimizerParamGroup> param_groups, ForeachAdamWOptions defaults = {});
  using torch::optim::Optimizer::step;
  torch::Tensor step(LossClosure closure = nullptr) override;

 private:
  int64_t step_count_ = 0;
};

}  // namespace olmo_cpp
