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

    // --- Norm weights must be checked FIRST (they appear inside lm_head, blocks, etc.) ---
    if (name.find("norm") != std::string::npos && p.dim() == 1) {
      p.fill_(1.0);

    // --- Embeddings: std=1.0 in µP (match both plain and multi-res names) ---
    } else if (name.find("embeddings") != std::string::npos ||
               name.find("token_embed") != std::string::npos ||
               name.find("role_embed") != std::string::npos ||
               name.find("char_embed") != std::string::npos ||
               name.find("phrase_embed") != std::string::npos) {
      double std_val = 1.0;
      trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);

    // --- LM head output weight: zero-init in µP ---
    } else if (mup_cfg.zero_init_lm_head &&
               name.find("lm_head") != std::string::npos &&
               name.find("w_out") != std::string::npos) {
      p.zero_();

    // --- Output projections (attention w_out, FFN w2): scale by width_ratio ---
    } else if (name.find("w_out") != std::string::npos ||
               name.find("w2") != std::string::npos) {
      double fan_in = p.dim() >= 2 ? static_cast<double>(p.size(1)) : static_cast<double>(p.size(0));
      double std_val = width_ratio / std::sqrt(fan_in);
      trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);

    // --- Projection layers (role_proj, char_proj, phrase_proj): 1/sqrt(fan_in) ---
    } else if (name.find("_proj") != std::string::npos) {
      double fan_in = p.dim() >= 2 ? static_cast<double>(p.size(1)) : static_cast<double>(p.size(0));
      double std_val = 1.0 / std::sqrt(fan_in);
      trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);

    // --- Input matrices (QKV, gate_up, w1, w3): standard 1/sqrt(fan_in) ---
    } else {
      double fan_in = p.dim() >= 2 ? static_cast<double>(p.size(1)) : static_cast<double>(p.size(0));
      double std_val = 1.0 / std::sqrt(fan_in);
      trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);
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
