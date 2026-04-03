#include "olmo_cpp/model/layer_norm_variants.hpp"

#include <cmath>

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

LayerNormImpl::LayerNormImpl(int64_t size, double eps, bool elementwise_affine, bool bias)
    : eps_(eps),
      elementwise_affine_(elementwise_affine),
      has_bias_(bias),
      size_(size) {
  if (elementwise_affine_) {
    weight_ = register_parameter("weight", torch::ones(size));
    if (has_bias_) {
      bias_ = register_parameter("bias", torch::zeros(size));
    }
  }
}

torch::Tensor LayerNormImpl::forward(torch::Tensor x) {
  // Compute mean and variance along the last dimension.
  auto mean = x.mean(-1, /*keepdim=*/true);
  auto var = x.var(-1, /*unbiased=*/false, /*keepdim=*/true);
  auto x_norm = (x - mean) / torch::sqrt(var + eps_);
  if (elementwise_affine_ && weight_.defined()) {
    x_norm = x_norm * weight_;
    if (has_bias_ && bias_.defined()) {
      x_norm = x_norm + bias_;
    }
  }
  return x_norm;
}

// ---------------------------------------------------------------------------
// L2Norm
// ---------------------------------------------------------------------------

L2NormImpl::L2NormImpl(int64_t size, double eps, bool elementwise_affine)
    : eps_(eps), elementwise_affine_(elementwise_affine) {
  if (elementwise_affine_) {
    weight_ = register_parameter("weight", torch::ones(size));
  }
}

torch::Tensor L2NormImpl::forward(torch::Tensor x) {
  // Compute L2 norm along the last dimension.
  auto norm = torch::norm(x, 2, /*dim=*/-1, /*keepdim=*/true);
  auto x_norm = x / torch::clamp_min(norm, eps_);
  if (elementwise_affine_ && weight_.defined()) {
    x_norm = x_norm * weight_;
  }
  return x_norm;
}

// ---------------------------------------------------------------------------
// FusedRMSNorm
// ---------------------------------------------------------------------------

FusedRMSNormImpl::FusedRMSNormImpl(int64_t size, double eps, bool elementwise_affine)
    : eps_(eps), elementwise_affine_(elementwise_affine), size_(size) {
  if (elementwise_affine_) {
    weight_ = register_parameter("weight", torch::ones(size));
  }
}

torch::Tensor FusedRMSNormImpl::forward(torch::Tensor x) {
  // RMS normalization: x * rsqrt(mean(x^2) + eps)
  // Only upcast to FP32 when input is a reduced-precision dtype
  if (x.dtype() == torch::kFloat32) {
    auto variance = x.pow(2).mean(-1, /*keepdim=*/true);
    auto x_norm = x * torch::rsqrt(variance + eps_);
    if (elementwise_affine_ && weight_.defined()) {
      x_norm = x_norm * weight_;
    }
    return x_norm;
  }

  // Reduced precision path: compute in FP32 for numerical stability
  auto input_dtype = x.dtype();
  auto x_fp32 = x.to(torch::kFloat32);
  auto variance = x_fp32.pow(2).mean(-1, /*keepdim=*/true);
  auto x_norm = x_fp32 * torch::rsqrt(variance + eps_);
  x_norm = x_norm.to(input_dtype);

  if (elementwise_affine_ && weight_.defined()) {
    x_norm = x_norm * weight_;
  }
  return x_norm;
}

}  // namespace olmo_cpp
