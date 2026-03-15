#include "olmo_cpp/model/lm_head.hpp"

namespace olmo_cpp {

LMHeadImpl::LMHeadImpl(int64_t d_model, int64_t vocab_size, bool use_norm, double eps)
    : w_out_(register_module("w_out", torch::nn::Linear(torch::nn::LinearOptions(d_model, vocab_size).bias(false)))) {
  if (use_norm) {
    norm_ = RMSNorm(d_model, eps);
    register_module("norm", *norm_);
  }
}

torch::Tensor LMHeadImpl::forward(torch::Tensor x) {
  if (norm_) {
    x = (*norm_)(x);
  }
  return w_out_(x);
}

}  // namespace olmo_cpp
