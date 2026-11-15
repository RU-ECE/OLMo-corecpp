/**
 * tools/eval_sft_loss.cpp — llm-cpp's main next-token loss on SFT windows, to compare with
 * Hugging Face on exactly the same tokens (scripts/hf_sft_loss.py prints the same number).
 *
 *   eval_sft_loss --config c.json --ckpt base.pt --tokens sft_tokens.npy --mask sft_mask.npy
 *       [--seq 4096] [--windows 8] [--bf16 0|1] [--dora 0|1] [--train 0|1] [--ckpt-act 0|1]
 *       [--device cuda|mps|cpu]
 *
 * Window k is tokens[k*S : k*S+S] with labels shifted by one and mask==0 labels ignored,
 * as olmo_train builds them. Prints each window's token-mean CE and their mean (what
 * olmo_train logs per micro-batch when there are no MTP heads). The switches reproduce
 * the trainer's setup one piece at a time: bf16 casts every float parameter and buffer
 * as main.cpp does, dora wraps the projections (magnitude initialised in train mode),
 * ckpt-act turns on full activation checkpointing.
 */
#include "olmo_cpp/backend/cuda_backend.hpp"
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"

#include <cnpy.h>
#include <torch/torch.h>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace olmo_cpp;

int main(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i + 1 < argc; i += 2) a[std::string(argv[i]).substr(2)] = argv[i + 1];
  auto get = [&](const std::string& k, const std::string& d = "") {
    auto it = a.find(k);
    if (it != a.end()) return it->second;
    if (d.empty()) { std::fprintf(stderr, "missing --%s\n", k.c_str()); std::exit(2); }
    return d;
  };
  const int64_t S = std::stoll(get("seq", "4096"));
  const int64_t n_win = std::stoll(get("windows", "8"));
  const bool bf16 = get("bf16", "0") == "1", dora = get("dora", "0") == "1";
  const bool train_mode = get("train", "0") == "1", ckpt_act = get("ckpt-act", "0") == "1";
  // grad: run the forward with autograd recording, as olmo_train does (activation
  // checkpointing only engages then). init: call init_weights() first, as main.cpp does.
  const bool with_grad = get("grad", "0") == "1", do_init = get("init", "0") == "1";
  // cuda-backend: what main.cpp does on CUDA — the olmo_ops kernels for RMSNorm,
  // SiLU-mul and RoPE, plus TF32 for fp32 matmuls.
  const bool cuda_backend = get("cuda-backend", "0") == "1";
  if (cuda_backend) {
    use_cuda_backend();
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
  }
  const std::string dev_s = get("device", "cuda");
  torch::Device dev = dev_s == "cuda" ? torch::Device(torch::kCUDA)
                    : dev_s == "mps"  ? torch::Device(torch::kMPS)
                                      : torch::Device(torch::kCPU);

  auto base_cfg = load_config_from_json(get("config"));
  base_cfg.num_mtp_heads = 0;
  base_cfg.validate();
  auto cfg = base_cfg;
  cfg.use_dora = dora;
  cfg.dora_rank = 32;
  cfg.dora_alpha = 32.0;
  if (ckpt_act) cfg.activation_checkpoint_mode = TransformerConfig::ActivationCheckpointMode::Full;
  Transformer model(cfg);
  if (do_init) model->init_weights();
  // Model to the device (and dtype) first, as main.cpp + train.cpp do: holding both a
  // CPU fp32 model and the CPU fp32 base needs 2 x 29 GB plus load buffers for the 7B.
  model->to(dev);
  if (bf16) {  // exactly main.cpp's pure-bf16 cast
    torch::NoGradGuard ng;
    for (auto& p : model->parameters())
      if (p.is_floating_point()) p.set_data(p.data().to(torch::kBFloat16));
    for (auto& b : model->buffers())
      if (b.is_floating_point()) b.set_data(b.to(torch::kBFloat16));
  }
  {
    Transformer base(base_cfg);
    torch::load(base, get("ckpt"), torch::kCPU);
    torch::NoGradGuard ng;
    auto dst = model->named_parameters();
    auto dbuf = model->named_buffers();
    int64_t np = 0, nb = 0;
    for (auto& it : base->named_parameters())
      if (auto* d = dst.find(it.key())) { d->copy_(it.value()); ++np; }
    for (auto& it : base->named_buffers())
      if (auto* d = dbuf.find(it.key())) { d->copy_(it.value()); ++nb; }
    std::fprintf(stderr, "copied %lld parameters, %lld buffers\n", (long long)np, (long long)nb);
  }
  if (train_mode) model->train(); else model->eval();
  if (with_grad && dora) {  // olmo_train freezes everything but the adapters
    for (auto& it : model->named_parameters())
      it.value().set_requires_grad(it.key().find("dora_") != std::string::npos);
  }

  cnpy::NpyArray ta = cnpy::npy_load(get("tokens")), ma = cnpy::npy_load(get("mask"));
  TORCH_CHECK(ta.word_size == 4 && ma.word_size == 1, "expected uint32 tokens and uint8 mask");
  auto toks = torch::from_blob(ta.data<uint32_t>(), {(int64_t)ta.num_vals}, torch::kInt32).to(torch::kInt64);
  auto mask = torch::from_blob(ma.data<uint8_t>(), {(int64_t)ma.num_vals}, torch::kUInt8).to(torch::kInt64);
  const int64_t avail = (toks.size(0) - 1) / S;

  c10::optional<torch::NoGradGuard> ng;
  if (!with_grad) ng.emplace();
  double sum = 0.0;
  int64_t n = 0;
  for (int64_t k = 0; k < std::min(n_win, avail); ++k) {
    auto x = toks.slice(0, k * S, k * S + S).unsqueeze(0).to(dev);
    auto y = toks.slice(0, k * S + 1, k * S + S + 1).clone();
    auto keep = mask.slice(0, k * S + 1, k * S + S + 1);
    if (keep.sum().item<int64_t>() == 0) continue;
    y.masked_fill_(keep == 0, -100);
    double l = model->forward(x, y.unsqueeze(0).to(dev), -100).item<double>();  // graph freed here
    std::printf("window %lld: main loss %.4f\n", (long long)k, l);
    sum += l;
    ++n;
  }
  std::printf("mean of per-window main loss = %.4f over %lld windows (bf16 %d, dora %d, train %d, ckpt-act %d, "
              "grad %d, init %d, cuda-backend %d, %s)\n",
              sum / n, (long long)n, bf16, dora, train_mode, ckpt_act, with_grad, do_init, cuda_backend,
              dev_s.c_str());
  return 0;
}
