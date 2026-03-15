#include "olmo_cpp/model/layer_norm.hpp"

namespace olmo_cpp {

RMSNormImpl::RMSNormImpl(int64_t size, double eps, bool elementwise_affine)
    : eps_(eps), elementwise_affine_(elementwise_affine) {
  if (elementwise_affine) {
    weight_ = register_parameter("weight", torch::ones(size));
  }
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
  auto variance = x.pow(2).mean(-1, /*keepdim=*/true);
  auto normed = x * torch::rsqrt(variance + eps_);
  if (elementwise_affine_ && weight_.defined()) {
    return normed * weight_;
  }
  return normed;
}

}  // namespace olmo_cpp
