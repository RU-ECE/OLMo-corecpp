#include "olmo_cpp/model/moe/moe.hpp"

namespace olmo_cpp {

MoELayerImpl::MoELayerImpl(int64_t d_model, int64_t hidden_size,
                           int64_t num_experts, int64_t top_k, bool dropless,
                           double capacity_factor, bool bias)
    : router_(register_module(
          "router", TopKRouter(d_model, num_experts, top_k))),
      num_experts_(num_experts),
      top_k_(top_k),
      dropless_(dropless) {
  if (dropless) {
    auto m = DroplessMoEMLP(d_model, hidden_size, num_experts, bias);
    register_module("mlp", m);
    mlp_ = torch::nn::AnyModule(std::move(m));
  } else {
    auto m = MoEMLP(d_model, hidden_size, num_experts, capacity_factor, bias);
    register_module("mlp", m);
    mlp_ = torch::nn::AnyModule(std::move(m));
  }
}

MoELayerImpl::MoEOutput MoELayerImpl::forward(torch::Tensor x,
                                               double zloss_weight,
                                               double lb_loss_weight) {
  // Remember original shape for reshaping back
  auto orig_shape = x.sizes().vec();
  auto d_model = x.size(-1);

  // Flatten to [B*S, D]
  x = x.reshape({-1, d_model});

  // Route tokens to experts
  auto [expert_weights, expert_indices, router_logits] = router_(x);

  // Process through expert MLPs
  torch::Tensor hidden_states;
  if (dropless_) {
    hidden_states = mlp_.get<DroplessMoEMLPImpl>().forward(
        x, expert_weights, expert_indices);
  } else {
    hidden_states = mlp_.get<MoEMLPImpl>().forward(
        x, expert_weights, expert_indices);
  }

  // Reshape back to original shape
  hidden_states = hidden_states.reshape(orig_shape);

  // Compute auxiliary loss
  auto aux_loss = MoELoss::auxiliary_loss(
      router_logits, expert_indices, num_experts_, zloss_weight, lb_loss_weight);

  return {hidden_states, router_logits, aux_loss};
}

}  // namespace olmo_cpp
