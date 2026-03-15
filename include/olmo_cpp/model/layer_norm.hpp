#pragma once

#include <torch/torch.h>

namespace olmo_cpp {

/// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
/// OLMo uses elementwise_affine (weight), no bias
class RMSNormImpl : public torch::nn::Module {
 public:
  RMSNormImpl(int64_t size, double eps = 1e-6, bool elementwise_affine = true);

  torch::Tensor forward(torch::Tensor x);

 private:
  torch::Tensor weight_;
  double eps_;
  bool elementwise_affine_;
};

TORCH_MODULE(RMSNorm);

}  // namespace olmo_cpp
