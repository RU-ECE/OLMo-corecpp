#include "olmo_cpp/model/gated_attention.hpp"

namespace olmo_cpp {

// --- HeadwiseGate ---

HeadwiseGateImpl::HeadwiseGateImpl(int64_t n_heads) {
  // Initialize to ones so sigmoid(gate_) starts near sigmoid(1) ~ 0.73
  gate_ = register_parameter("gate",
      torch::ones({1, n_heads, 1, 1}));
}

torch::Tensor HeadwiseGateImpl::forward(torch::Tensor x) {
  return torch::sigmoid(gate_) * x;
}

// --- ElementwiseGate ---

ElementwiseGateImpl::ElementwiseGateImpl(int64_t n_heads, int64_t head_dim) {
  // Initialize to zeros so sigmoid(gate_) starts at 0.5
  gate_ = register_parameter("gate",
      torch::zeros({1, n_heads, 1, head_dim}));
}

torch::Tensor ElementwiseGateImpl::forward(torch::Tensor x) {
  return torch::sigmoid(gate_) * x;
}

}  // namespace olmo_cpp
