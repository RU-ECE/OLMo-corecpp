#include "olmo_cpp/model/mup_init.hpp"
#include <torch/nn/init.h>
#include <iostream>
#include <cmath>

namespace {
void trunc_normal_(torch::Tensor& t, double mean, double std_val, double a, double b, torch::Generator gen) {
  t.normal_(mean, std_val, gen);
  t.clamp_(a, b);
}
}  // namespace

namespace olmo_cpp {

void apply_mup_init(
    torch::nn::Module& model,
    const TransformerConfig& cfg,
    const MuPConfig& mup_cfg,
    torch::optional<torch::Generator> gen) {

  torch::NoGradGuard no_grad;
  auto g = gen.value_or(torch::Generator());

  double target_width = mup_cfg.target_width > 0 ? mup_cfg.target_width : static_cast<double>(cfg.d_model);
  double width_ratio = mup_cfg.base_width / target_width;

  std::cout << "[µP] Initializing with base_width=" << mup_cfg.base_width
            << ", target_width=" << target_width
            << ", width_ratio=" << width_ratio << "\n";

  for (auto& named_param : model.named_parameters()) {
    auto& name = named_param.key();
    auto& p = named_param.value();

    if (!p.defined() || p.numel() == 0) continue;

    if (name.find("embeddings") != std::string::npos) {
      // Embedding: init with std=1.0 (not width-dependent in µP)
      double std_val = 1.0;
      trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);

    } else if (name.find("lm_head") != std::string::npos ||
               name.find("w_out") != std::string::npos) {
      if (mup_cfg.zero_init_lm_head &&
          name.find("lm_head") != std::string::npos &&
          name.find("norm") == std::string::npos) {
        // LM head output: zero-init in µP
        p.zero_();
      } else {
        // Output projections: scale by 1/d_model
        double std_val = 1.0 / target_width;
        trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);
      }

    } else if (name.find("norm") != std::string::npos) {
      // Normalization weights: init to 1.0 (standard)
      if (p.dim() == 1) {
        p.fill_(1.0);
      }

    } else {
      // Hidden layers: init with std = 1/sqrt(fan_in)
      // In µP, this is scaled by width_ratio for the hidden-to-hidden params
      double fan_in = p.dim() >= 2 ? static_cast<double>(p.size(1)) : static_cast<double>(p.size(0));
      double std_val = 1.0 / std::sqrt(fan_in);

      // Apply µP width scaling for hidden-to-hidden matrices
      if (name.find("w_q") != std::string::npos ||
          name.find("w_k") != std::string::npos ||
          name.find("w_v") != std::string::npos ||
          name.find("w_qkv") != std::string::npos ||
          name.find("w1") != std::string::npos ||
          name.find("w3") != std::string::npos ||
          name.find("w_gate_up") != std::string::npos) {
        // Input matrices: standard 1/sqrt(fan_in)
        trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);

      } else if (name.find("w2") != std::string::npos) {
        // Output matrices in FFN: scale by width_ratio
        std_val *= width_ratio;
        trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);

      } else {
        trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);
      }
    }
  }

  int64_t total_params = 0;
  for (auto& p : model.parameters()) total_params += p.numel();
  std::cout << "[µP] Initialized " << total_params << " parameters\n";
}

MuPLRMultiplier get_mup_lr_multipliers(
    const TransformerConfig& cfg,
    const MuPConfig& mup_cfg) {

  double target_width = mup_cfg.target_width > 0 ? mup_cfg.target_width : static_cast<double>(cfg.d_model);
  double width_ratio = mup_cfg.base_width / target_width;

  return MuPLRMultiplier{
    .embedding_mult = 1.0,           // Embeddings: standard LR
    .hidden_mult = width_ratio,      // Hidden: scale down LR with width
    .output_mult = width_ratio,      // Output: also scale down
  };
}

}  // namespace olmo_cpp
