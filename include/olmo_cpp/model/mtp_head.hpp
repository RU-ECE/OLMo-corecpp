#pragma once

#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/lm_head.hpp"
#include <torch/torch.h>

namespace olmo_cpp {

/// Single MTP prediction head: Linear projection + RMSNorm
/// Shares the LM head's output projection with the main model.
///
/// For head k predicting token at position t+k+1:
///   logits_k = shared_lm_head( norm_k( proj_k(h_t) ) )
///
/// During training, the loss target is labels shifted by k positions.
class MTPHeadImpl : public torch::nn::Module {
 public:
  MTPHeadImpl(int64_t d_model, double eps = 1e-6);

  /// Project hidden states through this head's transform.
  /// Returns transformed hidden states (caller applies shared LM head).
  torch::Tensor forward(torch::Tensor hidden_states);

 private:
  torch::nn::Linear proj_;
  RMSNorm norm_;
};

TORCH_MODULE(MTPHead);

}  // namespace olmo_cpp
