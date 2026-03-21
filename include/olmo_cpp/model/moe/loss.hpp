#pragma once

#include <torch/torch.h>

namespace olmo_cpp {

/// MoE auxiliary losses for load balancing
struct MoELoss {
  /// Z-loss: encourages router logits to stay small for stability
  /// loss = mean(logsumexp(logits, dim=-1)^2)
  static torch::Tensor z_loss(const torch::Tensor& router_logits);

  /// Load balancing loss: encourages uniform expert utilization
  /// loss = num_experts * sum(f_i * p_i) where f_i=fraction of tokens to expert i,
  /// p_i=average routing probability for expert i
  static torch::Tensor load_balancing_loss(
      const torch::Tensor& router_logits,
      const torch::Tensor& expert_indices,
      int64_t num_experts);

  /// Combined auxiliary loss
  static torch::Tensor auxiliary_loss(
      const torch::Tensor& router_logits,
      const torch::Tensor& expert_indices,
      int64_t num_experts,
      double zloss_weight = 1e-3,
      double lb_loss_weight = 1e-2);
};

}  // namespace olmo_cpp
