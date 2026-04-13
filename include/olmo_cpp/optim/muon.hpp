#pragma once
#include <torch/torch.h>
#include <vector>
#include <unordered_map>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#endif

namespace olmo_cpp {

struct MuonOptions : public torch::optim::OptimizerCloneableOptions<MuonOptions> {
  MuonOptions(double lr = 0.02) : lr_(lr) {}
  TORCH_ARG(double, lr) = 0.02;
  TORCH_ARG(double, momentum) = 0.95;
  TORCH_ARG(double, weight_decay) = 0.0;
  TORCH_ARG(int64_t, ns_steps) = 5;
  TORCH_ARG(bool, async_ns) = false;
  TORCH_ARG(int64_t, async_min_numel) = 65536;
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

#ifdef USE_CUDA
  struct AsyncState {
    torch::Tensor pending_update;
    at::cuda::CUDAEvent ready_event;
    bool has_pending = false;
  };
  std::unordered_map<void*, AsyncState> async_states_;
  c10::optional<at::cuda::CUDAStream> ns_stream_;
#endif
};

}  // namespace olmo_cpp
