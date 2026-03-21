#pragma once
#include <torch/torch.h>

namespace olmo_cpp {

/// Gated attention: multiplies attention output by a learned gate
/// Headwise: one scalar gate per head [n_heads]
/// Elementwise: one gate per element [n_heads, head_dim]
class HeadwiseGateImpl : public torch::nn::Module {
 public:
  HeadwiseGateImpl(int64_t n_heads);
  /// x: [B, n_heads, S, head_dim] -> same shape
  torch::Tensor forward(torch::Tensor x);
 private:
  torch::Tensor gate_;  // [1, n_heads, 1, 1]
};
TORCH_MODULE(HeadwiseGate);

class ElementwiseGateImpl : public torch::nn::Module {
 public:
  ElementwiseGateImpl(int64_t n_heads, int64_t head_dim);
  torch::Tensor forward(torch::Tensor x);
 private:
  torch::Tensor gate_;  // [1, n_heads, 1, head_dim]
};
TORCH_MODULE(ElementwiseGate);

}  // namespace olmo_cpp
