#pragma once

#include <torch/torch.h>

namespace olmo_cpp {

/// Standard LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
class LayerNormImpl : public torch::nn::Module {
 public:
  LayerNormImpl(int64_t size, double eps = 1e-5, bool elementwise_affine = true, bool bias = true);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::Tensor weight_, bias_;
  double eps_;
  bool elementwise_affine_, has_bias_;
  int64_t size_;
};
TORCH_MODULE(LayerNorm);

/// L2Norm: y = x / ||x||_2 * weight
class L2NormImpl : public torch::nn::Module {
 public:
  L2NormImpl(int64_t size, double eps = 1e-6, bool elementwise_affine = true);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::Tensor weight_;
  double eps_;
  bool elementwise_affine_;
};
TORCH_MODULE(L2Norm);

/// FusedRMSNorm: same as RMSNorm but with a hint for fused kernels
/// When CUDA kernels are available, uses fused implementation
class FusedRMSNormImpl : public torch::nn::Module {
 public:
  FusedRMSNormImpl(int64_t size, double eps = 1e-6, bool elementwise_affine = true);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::Tensor weight_;
  double eps_;
  bool elementwise_affine_;
  int64_t size_;
};
TORCH_MODULE(FusedRMSNorm);

}  // namespace olmo_cpp
