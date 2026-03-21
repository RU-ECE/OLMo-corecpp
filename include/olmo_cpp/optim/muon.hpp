#pragma once
#include <torch/torch.h>

namespace olmo_cpp {

/// Muon optimizer: momentum + orthogonalization via Newton-Schulz iteration
struct MuonOptions : public torch::optim::OptimizerCloneableOptions<MuonOptions> {
  MuonOptions(double lr = 0.02) : lr_(lr) {}
  TORCH_ARG(double, lr) = 0.02;
  TORCH_ARG(double, momentum) = 0.95;
  TORCH_ARG(double, weight_decay) = 0.0;
  TORCH_ARG(int64_t, ns_steps) = 5;  // Newton-Schulz iterations
  void set_lr(double lr) override { lr_ = lr; }
  double get_lr() const override { return lr_; }
};

class Muon : public torch::optim::Optimizer {
 public:
  explicit Muon(std::vector<torch::optim::OptimizerParamGroup> param_groups, MuonOptions defaults = {});
  explicit Muon(std::vector<torch::Tensor> params, MuonOptions defaults = {});
  using torch::optim::Optimizer::step;
  torch::Tensor step(LossClosure closure = nullptr) override;
 private:
  static torch::Tensor newton_schulz_orthogonalize(torch::Tensor G, int64_t steps);
};

}  // namespace olmo_cpp
