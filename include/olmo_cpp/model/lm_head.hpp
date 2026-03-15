#pragma once

#include "olmo_cpp/model/layer_norm.hpp"
#include <torch/torch.h>

namespace olmo_cpp {

/// LM Head: optional RMSNorm + Linear to vocab
class LMHeadImpl : public torch::nn::Module {
 public:
  LMHeadImpl(int64_t d_model, int64_t vocab_size, bool use_norm = true, double eps = 1e-6);

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear w_out() { return w_out_; }

 private:
  std::optional<RMSNorm> norm_;
  torch::nn::Linear w_out_;
};

TORCH_MODULE(LMHead);

}  // namespace olmo_cpp
