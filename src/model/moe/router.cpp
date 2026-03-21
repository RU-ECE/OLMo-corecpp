#include "olmo_cpp/model/moe/router.hpp"

namespace olmo_cpp {

TopKRouterImpl::TopKRouterImpl(int64_t d_model, int64_t num_experts,
                               int64_t top_k, bool normalize_weights)
    : gate_(register_module(
          "gate",
          torch::nn::Linear(
              torch::nn::LinearOptions(d_model, num_experts).bias(false)))),
      num_experts_(num_experts),
      top_k_(top_k),
      normalize_weights_(normalize_weights) {}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
TopKRouterImpl::forward(torch::Tensor x) {
  // Flatten to [B*S, D]
  auto d_model = x.size(-1);
  x = x.reshape({-1, d_model});

  // Compute router logits: [B*S, num_experts]
  auto logits = gate_(x);

  // Select top-k experts per token
  auto [top_values, top_indices] = logits.topk(top_k_, /*dim=*/-1);

  // Compute softmax weights over the selected top-k logits
  auto weights = torch::softmax(top_values, /*dim=*/-1);

  // Optionally normalize so weights sum to 1 (they already do after softmax,
  // but this is kept explicit for clarity and in case of future masking)
  if (normalize_weights_) {
    weights = weights / weights.sum(/*dim=*/-1, /*keepdim=*/true);
  }

  return {weights, top_indices, logits};
}

}  // namespace olmo_cpp
