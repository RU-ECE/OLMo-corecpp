/**
 * tools/mps_diag.cpp — where does a real checkpoint diverge between CPU and MPS?
 *
 * Runs the same token ids through one model on CPU and one on MPS and prints,
 * per sequence length, the max |diff| of the final logits, whether the last
 * position's argmax agrees, and the first block whose output diverges.
 *
 *   mps_diag --config c.json --ckpt m.pt [--lens 16,64,512] [--offset 1000]
 */
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"

#include <torch/torch.h>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace olmo_cpp;

int main(int argc, char** argv) {
  std::string cfg_path, ckpt, lens = "16,64,512";
  int64_t offset = 1000;
  for (int i = 1; i + 1 < argc; ++i) {
    std::string a = argv[i];
    if (a == "--config") cfg_path = argv[++i];
    else if (a == "--ckpt") ckpt = argv[++i];
    else if (a == "--lens") lens = argv[++i];
    else if (a == "--offset") offset = std::stoll(argv[++i]);
  }
  auto cfg = load_config_from_json(cfg_path);
  cfg.validate();
  Transformer cpu(cfg), mps(cfg);
  torch::load(cpu, ckpt, torch::kCPU);
  torch::load(mps, ckpt, torch::kCPU);
  mps->to(torch::kMPS);
  cpu->eval();
  mps->eval();
  torch::NoGradGuard ng;
  std::stringstream ss(lens);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    const int64_t S = std::stoll(tok);
    auto x = torch::arange(offset, offset + S, torch::kInt64).remainder(cfg.vocab_size).unsqueeze(0);
    auto hc = cpu->forward_backbone(x);
    auto hm = mps->forward_backbone(x.to(torch::kMPS)).to(torch::kCPU);
    auto lc = cpu->apply_lm_head(hc), lm = mps->apply_lm_head(hm.to(torch::kMPS)).to(torch::kCPU);
    const double dh = (hc - hm).abs().max().item<double>(), sh = hc.abs().max().item<double>();
    const double dl = (lc - lm).abs().max().item<double>(), sl = lc.abs().max().item<double>();
    const bool same = lc.select(1, S - 1).argmax().item<int64_t>() == lm.select(1, S - 1).argmax().item<int64_t>();
    std::printf("S=%4lld  hidden max|diff| %.3e (scale %.2e)  logits max|diff| %.3e (scale %.2e)  last argmax %s\n",
                (long long)S, dh, sh, dl, sl, same ? "same" : "DIFFERENT");
    // Per position: where does it break?
    auto per_pos = (hc - hm).abs().amax(-1).squeeze(0);  // [S]
    int64_t first_bad = -1;
    for (int64_t t = 0; t < S; ++t)
      if (per_pos[t].item<double>() > 1e-2 * sh) { first_bad = t; break; }
    std::printf("        first position with hidden diff > 1%% of scale: %lld\n", (long long)first_bad);
  }
  return 0;
}
