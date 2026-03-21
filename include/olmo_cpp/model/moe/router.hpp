#pragma once

#include <torch/torch.h>
#include <tuple>

namespace olmo_cpp {

/// Top-K router: routes each token to top-k experts
/// Returns (expert_weights, expert_indices, router_logits)
class TopKRouterImpl : public torch::nn::Module {
 public:
  TopKRouterImpl(int64_t d_model, int64_t num_experts, int64_t top_k,
                 bool normalize_weights = true);

  /// Returns: {expert_weights [B*S, top_k], expert_indices [B*S, top_k], logits [B*S, E]}
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);

 private:
  torch::nn::Linear gate_;
  int64_t num_experts_, top_k_;
  bool normalize_weights_;
};

TORCH_MODULE(TopKRouter);

}  // namespace olmo_cpp
