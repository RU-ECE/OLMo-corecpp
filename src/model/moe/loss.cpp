#include "olmo_cpp/model/moe/loss.hpp"

namespace olmo_cpp {

torch::Tensor MoELoss::z_loss(const torch::Tensor& router_logits) {
  // router_logits: [B*S, num_experts]
  // loss = mean(logsumexp(logits, dim=-1)^2)
  auto lse = torch::logsumexp(router_logits, /*dim=*/-1);  // [B*S]
  return (lse * lse).mean();
}

torch::Tensor MoELoss::load_balancing_loss(
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_indices,
    int64_t num_experts) {
  // router_logits: [B*S, num_experts]
  // expert_indices: [B*S, top_k]
  auto num_tokens = router_logits.size(0);
  auto top_k = expert_indices.size(1);

  // f_i: fraction of tokens routed to each expert
  // Create one-hot over all top-k slots and sum across the top_k dimension
  auto flat_indices = expert_indices.reshape({-1});  // [B*S * top_k]
  auto one_hot = torch::one_hot(flat_indices, num_experts).to(router_logits.dtype());
  // [B*S * top_k, num_experts] -> sum to get counts per expert
  auto expert_counts = one_hot.sum(/*dim=*/0);  // [num_experts]
  auto f = expert_counts / static_cast<double>(num_tokens * top_k);  // [num_experts]

  // p_i: average routing probability for each expert
  auto probs = torch::softmax(router_logits, /*dim=*/-1);  // [B*S, num_experts]
  auto p = probs.mean(/*dim=*/0);  // [num_experts]

  // loss = num_experts * dot(f, p)
  return static_cast<double>(num_experts) * (f * p).sum();
}

torch::Tensor MoELoss::auxiliary_loss(
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_indices,
    int64_t num_experts,
    double zloss_weight,
    double lb_loss_weight) {
  auto zl = z_loss(router_logits);
  auto lbl = load_balancing_loss(router_logits, expert_indices, num_experts);
  return zloss_weight * zl + lb_loss_weight * lbl;
}

}  // namespace olmo_cpp
