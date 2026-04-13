#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/backend/backend.hpp"

namespace olmo_cpp {

RMSNormImpl::RMSNormImpl(int64_t size, double eps, bool elementwise_affine)
    : eps_(eps), elementwise_affine_(elementwise_affine) {
  if (elementwise_affine) {
    weight_ = register_parameter("weight", torch::ones(size));
  }
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
  torch::Tensor w = (elementwise_affine_ && weight_.defined()) ? weight_ : torch::Tensor();
  return get_backend().rms_norm(x, w, eps_);
}

torch::Tensor RMSNormImpl::forward_add(torch::Tensor x, torch::Tensor residual) {
  torch::Tensor w = (elementwise_affine_ && weight_.defined()) ? weight_ : torch::Tensor();
  return get_backend().rms_norm_add(x, residual, w, eps_);
}

}  // namespace olmo_cpp
