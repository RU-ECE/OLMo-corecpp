#include "olmo_cpp/model/moe/mlp.hpp"

#include <algorithm>

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// ExpertMLP (SwiGLU, identical to FeedForward)
// ---------------------------------------------------------------------------

ExpertMLPImpl::ExpertMLPImpl(int64_t d_model, int64_t hidden_size, bool bias)
    : w1_(register_module(
          "w1",
          torch::nn::Linear(
              torch::nn::LinearOptions(d_model, hidden_size).bias(bias)))),
      w2_(register_module(
          "w2",
          torch::nn::Linear(
              torch::nn::LinearOptions(hidden_size, d_model).bias(bias)))),
      w3_(register_module(
          "w3",
          torch::nn::Linear(
              torch::nn::LinearOptions(d_model, hidden_size).bias(bias)))) {}

torch::Tensor ExpertMLPImpl::forward(torch::Tensor x) {
  return w2_(torch::silu(w1_(x)) * w3_(x));
}

// ---------------------------------------------------------------------------
// MoEMLP (with capacity factor)
// ---------------------------------------------------------------------------

MoEMLPImpl::MoEMLPImpl(int64_t d_model, int64_t hidden_size,
                        int64_t num_experts, double capacity_factor, bool bias)
    : experts_(register_module("experts", torch::nn::ModuleList())),
      num_experts_(num_experts),
      capacity_factor_(capacity_factor) {
  for (int64_t i = 0; i < num_experts; ++i) {
    experts_->push_back(ExpertMLP(d_model, hidden_size, bias));
  }
}

torch::Tensor MoEMLPImpl::forward(torch::Tensor x,
                                   torch::Tensor expert_weights,
                                   torch::Tensor expert_indices) {
  // x: [B*S, D], expert_weights: [B*S, K], expert_indices: [B*S, K]
  auto num_tokens = x.size(0);
  auto d_model = x.size(1);
  auto top_k = expert_indices.size(1);

  // Capacity: maximum number of tokens any single expert can process
  auto capacity = static_cast<int64_t>(
      std::ceil(static_cast<double>(num_tokens * top_k) /
                static_cast<double>(num_experts_) * capacity_factor_));

  auto output = torch::zeros_like(x);  // [B*S, D]

  for (int64_t e = 0; e < num_experts_; ++e) {
    // Find all (token, slot) pairs where expert_indices == e
    auto mask = expert_indices.eq(e);  // [B*S, K]

    // Get token indices and slot indices where this expert is selected
    auto positions = mask.nonzero();  // [N_assigned, 2]

    if (positions.size(0) == 0) {
      continue;
    }

    // Apply capacity limit: only process up to `capacity` tokens
    auto n_assigned = positions.size(0);
    auto n_process = std::min(n_assigned, capacity);
    positions = positions.slice(/*dim=*/0, /*start=*/0, /*end=*/n_process);

    auto token_ids = positions.select(/*dim=*/1, /*index=*/0);  // [N]
    auto slot_ids = positions.select(/*dim=*/1, /*index=*/1);   // [N]

    // Gather input tokens
    auto expert_input = x.index_select(/*dim=*/0, token_ids);  // [N, D]

    // Process through expert
    auto expert_output =
        experts_[e]->as<ExpertMLPImpl>()->forward(expert_input);  // [N, D]

    // Gather the corresponding weights
    auto w = expert_weights.index({token_ids, slot_ids}).unsqueeze(1);  // [N, 1]

    // Scatter-add weighted expert output back to output tensor
    output.index_add_(/*dim=*/0, token_ids, expert_output * w);
  }

  return output;
}

// ---------------------------------------------------------------------------
// DroplessMoEMLP (no token dropping)
// ---------------------------------------------------------------------------

DroplessMoEMLPImpl::DroplessMoEMLPImpl(int64_t d_model, int64_t hidden_size,
                                       int64_t num_experts, bool bias)
    : experts_(register_module("experts", torch::nn::ModuleList())),
      num_experts_(num_experts) {
  for (int64_t i = 0; i < num_experts; ++i) {
    experts_->push_back(ExpertMLP(d_model, hidden_size, bias));
  }
}

torch::Tensor DroplessMoEMLPImpl::forward(torch::Tensor x,
                                           torch::Tensor expert_weights,
                                           torch::Tensor expert_indices) {
  // x: [B*S, D], expert_weights: [B*S, K], expert_indices: [B*S, K]
  auto output = torch::zeros_like(x);  // [B*S, D]

  for (int64_t e = 0; e < num_experts_; ++e) {
    // Find all (token, slot) pairs where expert_indices == e
    auto mask = expert_indices.eq(e);  // [B*S, K]
    auto positions = mask.nonzero();   // [N_assigned, 2]

    if (positions.size(0) == 0) {
      continue;
    }

    auto token_ids = positions.select(/*dim=*/1, /*index=*/0);  // [N]
    auto slot_ids = positions.select(/*dim=*/1, /*index=*/1);   // [N]

    // Gather input tokens
    auto expert_input = x.index_select(/*dim=*/0, token_ids);  // [N, D]

    // Process through expert (no capacity limit)
    auto expert_output =
        experts_[e]->as<ExpertMLPImpl>()->forward(expert_input);  // [N, D]

    // Gather the corresponding weights
    auto w = expert_weights.index({token_ids, slot_ids}).unsqueeze(1);  // [N, 1]

    // Scatter-add weighted expert output back to output tensor
    output.index_add_(/*dim=*/0, token_ids, expert_output * w);
  }

  return output;
}

}  // namespace olmo_cpp
