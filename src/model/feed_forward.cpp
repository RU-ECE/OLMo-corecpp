#include "olmo_cpp/model/feed_forward.hpp"
#include "olmo_cpp/backend/backend.hpp"

namespace olmo_cpp {

FeedForwardImpl::FeedForwardImpl(int64_t d_model, int64_t hidden_size, bool bias,
                                  bool use_fused_gate_up)
    : fused_(use_fused_gate_up) {
  if (fused_) {
    w_gate_up_ = register_module("w_gate_up",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, 2 * hidden_size).bias(bias)));
  } else {
    w1_ = register_module("w1",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, hidden_size).bias(bias)));
    w3_ = register_module("w3",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, hidden_size).bias(bias)));
  }
  w2_ = register_module("w2",
      torch::nn::Linear(torch::nn::LinearOptions(hidden_size, d_model).bias(bias)));
}

torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
  if (fused_) {
    // Single GEMM for gate+up, then split
    auto gate_up = w_gate_up_(x);
    int64_t h = gate_up.size(-1) / 2;
    auto gate = gate_up.narrow(-1, 0, h);
    auto up = gate_up.narrow(-1, h, h);
    return w2_(get_backend().silu_mul(gate, up));
  }
  return w2_(get_backend().silu_mul(w1_(x), w3_(x)));
}

}  // namespace olmo_cpp
