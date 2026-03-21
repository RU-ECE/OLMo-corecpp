#include "olmo_cpp/model/mtp_head.hpp"

namespace olmo_cpp {

MTPHeadImpl::MTPHeadImpl(int64_t d_model, double eps)
    : proj_(register_module("proj",
          torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model).bias(false)))),
      norm_(register_module("norm", RMSNorm(d_model, eps))) {}

torch::Tensor MTPHeadImpl::forward(torch::Tensor hidden_states) {
  return norm_(proj_(hidden_states));
}

}  // namespace olmo_cpp
