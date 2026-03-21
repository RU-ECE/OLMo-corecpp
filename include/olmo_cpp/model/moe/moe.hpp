#pragma once

#include "olmo_cpp/model/moe/router.hpp"
#include "olmo_cpp/model/moe/mlp.hpp"
#include "olmo_cpp/model/moe/loss.hpp"
#include <torch/torch.h>

namespace olmo_cpp {

/// Full MoE layer: router + expert MLPs + auxiliary loss
class MoELayerImpl : public torch::nn::Module {
 public:
  MoELayerImpl(int64_t d_model, int64_t hidden_size, int64_t num_experts,
               int64_t top_k = 2, bool dropless = true,
               double capacity_factor = 1.25, bool bias = false);

  /// Returns output tensor. Accumulates aux_loss in the returned struct.
  struct MoEOutput {
    torch::Tensor hidden_states;
    torch::Tensor router_logits;  // for computing aux loss externally
    torch::Tensor aux_loss;       // pre-computed if requested
  };

  MoEOutput forward(torch::Tensor x, double zloss_weight = 1e-3,
                    double lb_loss_weight = 1e-2);

 private:
  TopKRouter router_;
  torch::nn::AnyModule mlp_;  // Either MoEMLP or DroplessMoEMLP
  int64_t num_experts_, top_k_;
  bool dropless_;
};

TORCH_MODULE(MoELayer);

}  // namespace olmo_cpp
