#pragma once

#include <torch/torch.h>

namespace olmo_cpp {

/// SwiGLU feed-forward: out = w2(silu(w1(x)) * w3(x))
/// When use_fused_gate_up is true, w1 and w3 are combined into a single
/// [2*H, D] weight matrix for a single GEMM (halves launch overhead).
class FeedForwardImpl : public torch::nn::Module {
 public:
  FeedForwardImpl(int64_t d_model, int64_t hidden_size, bool bias = false,
                  bool use_fused_gate_up = false);

  torch::Tensor forward(torch::Tensor x);

 private:
  // Standard path: separate gate (w1) and up (w3)
  torch::nn::Linear w1_{nullptr};
  torch::nn::Linear w3_{nullptr};

  // Fused path: combined gate+up weight [2*H, D]
  torch::nn::Linear w_gate_up_{nullptr};

  // Down projection (always separate)
  torch::nn::Linear w2_{nullptr};

  bool fused_ = false;
};

TORCH_MODULE(FeedForward);

}  // namespace olmo_cpp
