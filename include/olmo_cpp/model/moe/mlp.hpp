#pragma once

#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

/// Single expert MLP (SwiGLU): out = w2(silu(w1(x)) * w3(x))
class ExpertMLPImpl : public torch::nn::Module {
 public:
  ExpertMLPImpl(int64_t d_model, int64_t hidden_size, bool bias = false);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Linear w1_, w2_, w3_;
};

TORCH_MODULE(ExpertMLP);

/// MoE MLP: collection of expert MLPs with capacity factor
class MoEMLPImpl : public torch::nn::Module {
 public:
  MoEMLPImpl(int64_t d_model, int64_t hidden_size, int64_t num_experts,
             double capacity_factor = 1.25, bool bias = false);

  /// x: [B*S, D], expert_weights: [B*S, K], expert_indices: [B*S, K]
  torch::Tensor forward(torch::Tensor x, torch::Tensor expert_weights,
                        torch::Tensor expert_indices);

 private:
  torch::nn::ModuleList experts_;
  int64_t num_experts_;
  double capacity_factor_;
};

TORCH_MODULE(MoEMLP);

/// Dropless MoE MLP: no token dropping, processes all assigned tokens
class DroplessMoEMLPImpl : public torch::nn::Module {
 public:
  DroplessMoEMLPImpl(int64_t d_model, int64_t hidden_size, int64_t num_experts,
                     bool bias = false);

  torch::Tensor forward(torch::Tensor x, torch::Tensor expert_weights,
                        torch::Tensor expert_indices);

 private:
  torch::nn::ModuleList experts_;
  int64_t num_experts_;
};

TORCH_MODULE(DroplessMoEMLP);

}  // namespace olmo_cpp
