/**
 * tools/test_dora_init.cpp — a DoRA finetune must start exactly at the base model.
 *
 * Built the way olmo_train builds it (main.cpp: construct, init_weights(); train.cpp:
 * copy the base parameters by name, train mode so the DoRA magnitude lazy-initialises),
 * the DoRA model's logits must equal the base model's. init_weights() used to give every
 * 2-D block parameter N(0, init_std), lora_B included, so W + s*B*A != W from step 0:
 * on OLMo-2-1124-7B the main loss on the Forge SFT windows was 6.5 instead of 1.03.
 *
 *   ./build/test_dora_init
 */
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"

#include <torch/torch.h>
#include <cstdio>
#include <string>

using namespace olmo_cpp;

int main() {
  int fail = 0;
  TransformerConfig c;
  c.d_model = 64;
  c.n_layers = 2;
  c.n_heads = 4;
  c.vocab_size = 101;
  c.hidden_size_multiple_of = 32;
  c.use_qk_norm = true;
  c.init_std = 0.02;

  torch::manual_seed(0);
  Transformer base(c);
  base->init_weights();
  base->to(torch::kFloat64);

  auto dc = c;
  dc.use_dora = true;
  dc.dora_rank = 4;
  dc.dora_alpha = 4.0;
  Transformer ft(dc);
  ft->init_weights();  // main.cpp calls this for every run, finetunes included
  ft->to(torch::kFloat64);
  {
    torch::NoGradGuard ng;
    auto dst = ft->named_parameters();
    for (auto& it : base->named_parameters()) {
      auto* d = dst.find(it.key());
      TORCH_CHECK(d, "DoRA model has no ", it.key());
      d->copy_(it.value());
    }
  }
  double max_b = 0.0;
  for (auto& it : ft->named_parameters())
    if (it.key().find("lora_B") != std::string::npos)
      max_b = std::max(max_b, it.value().abs().max().item<double>());

  base->eval();
  ft->train();  // DoRA magnitude initialises from the loaded base on this first forward
  auto x = torch::randint(0, c.vocab_size, {2, 24}, torch::kInt64);
  torch::NoGradGuard ng;
  auto lb = base->forward(x);
  auto lf = ft->forward(x);
  const double rel = (lf - lb).abs().max().item<double>() / lb.abs().max().item<double>();

  char b[200];
  std::snprintf(b, sizeof b, "every lora_B is zero after init_weights (max |B| = %.2e)", max_b);
  std::printf("%s  %s\n", max_b == 0.0 ? "PASS" : "FAIL", b);
  fail += max_b != 0.0;
  // DoRA forms the row norm in fp32 by design, so agreement is to fp32 precision.
  std::snprintf(b, sizeof b, "DoRA model at step 0 == base model (logits rel diff %.2e)", rel);
  std::printf("%s  %s\n", rel < 1e-5 ? "PASS" : "FAIL", b);
  fail += rel >= 1e-5;
  std::printf("\n%s: %d failure(s)\n", fail ? "FAILED" : "ALL PASS", fail);
  return fail ? 1 : 0;
}
