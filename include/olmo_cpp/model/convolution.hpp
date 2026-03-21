#pragma once

#include <torch/torch.h>

namespace olmo_cpp {

/// 1D causal convolution for hybrid architectures (Mamba-like)
class CausalConv1dImpl : public torch::nn::Module {
 public:
  CausalConv1dImpl(int64_t channels, int64_t kernel_size);

  /// x: [B, S, D] -> [B, S, D] (causal: only looks at past)
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Conv1d conv_;
  int64_t padding_;
};
TORCH_MODULE(CausalConv1d);

}  // namespace olmo_cpp
