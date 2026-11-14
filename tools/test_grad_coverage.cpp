/**
 * tools/test_grad_coverage.cpp — proof test for speed-plan bugs 1 and 4.
 *
 * Every parameter that requires a gradient must receive a non-zero one after
 * a training forward + backward. A fast path that writes through raw
 * pointers or a library GEMM into a fresh tensor (cuBLASLt fast_linear on
 * CUDA = bug 1, SIMD CPU kernels = bug 4) silently cuts the graph: the step
 * still runs, the loss still falls (through the residual stream), and the
 * cut weights never train.
 *
 * Runs on each available device with that device's training backend
 * (SIMD on CPU, CUDA backend on CUDA, default on MPS), for:
 *   - dense pretraining with MTP heads (every parameter trains);
 *   - DoRA finetuning (base frozen; dora_* adapters and MTP heads train),
 *     with lora_B made non-zero so the adapters' A matrices get gradients.
 * Prints every parameter that is missing a gradient. Exit 0 = all pass.
 *
 *   ./build/test_grad_coverage [cpu|mps|cuda ...]   (default: all available)
 */
#include "olmo_cpp/backend/backend.hpp"
#include "olmo_cpp/backend/simd_backend.hpp"
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/backend/cuda_backend.hpp"

#include <torch/torch.h>
#include <cstdio>
#include <string>
#include <vector>

using namespace olmo_cpp;

namespace {

TransformerConfig cfg_for(bool dora) {
  TransformerConfig c;
  c.d_model = 64;
  c.n_layers = 2;
  c.n_heads = 4;
  c.n_kv_heads = 2;
  c.vocab_size = 257;
  c.hidden_size_multiple_of = 32;
  c.num_mtp_heads = 2;
  c.use_dora = dora;
  c.dora_rank = 4;
  return c;
}

// Returns the number of trainable parameters without a usable gradient.
int run_case(const std::string& name, torch::Device device, bool dora) {
  torch::manual_seed(0);
  auto cfg = cfg_for(dora);
  Transformer model(cfg);
  model->init_weights();
  model->to(device);
  model->train();
  int64_t n_trainable = 0;
  for (auto& it : model->named_parameters()) {
    bool trainable = true;
    if (dora) {
      trainable = it.key().find("dora_") != std::string::npos ||
                  it.key().find("mtp_heads") != std::string::npos;  // as src/train.cpp
      if (trainable && it.key().find("lora_B") != std::string::npos) {
        torch::NoGradGuard ng;
        it.value().normal_(0.0, 0.02);  // zero B gives A a zero gradient by design
      }
    }
    it.value().set_requires_grad(trainable);
    n_trainable += trainable ? 1 : 0;
  }

  const int64_t B = 2, S = 16;
  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto input = torch::randint(0, cfg.vocab_size, {B, S}, opts);
  auto labels = torch::randint(0, cfg.vocab_size, {B, S}, opts);
  auto loss = model->forward(input, labels, -100);
  loss.backward();

  std::vector<std::string> missing;
  for (auto& it : model->named_parameters()) {
    if (!it.value().requires_grad()) continue;
    const auto& g = it.value().grad();
    if (!g.defined()) missing.push_back(it.key() + " (no grad)");
    else if (g.abs().sum().item<double>() == 0.0) missing.push_back(it.key() + " (zero grad)");
  }
  std::printf("%s  %s: %lld/%lld trainable parameters got a gradient (loss %.4f)\n",
              missing.empty() ? "PASS" : "FAIL", name.c_str(),
              static_cast<long long>(n_trainable - static_cast<int64_t>(missing.size())),
              static_cast<long long>(n_trainable), loss.item<double>());
  for (const auto& m : missing) std::printf("        missing: %s\n", m.c_str());
  return static_cast<int>(missing.size());
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> devices;
  for (int i = 1; i < argc; ++i) devices.emplace_back(argv[i]);
  if (devices.empty()) {
    devices.push_back("cpu");
    if (torch::mps::is_available()) devices.push_back("mps");
    if (torch::cuda::is_available()) devices.push_back("cuda");
  }
  int fails = 0;
  for (const auto& d : devices) {
    torch::Device device = d == "cuda" ? torch::Device(torch::kCUDA)
                         : d == "mps"  ? torch::Device(torch::kMPS)
                                       : torch::Device(torch::kCPU);
    if (d == "cpu") use_simd_backend();
    else if (d == "cuda") use_cuda_backend();
    else set_backend(nullptr);  // default LibTorch backend
    fails += run_case(d + " dense+MTP", device, /*dora=*/false) ? 1 : 0;
    fails += run_case(d + " DoRA+MTP", device, /*dora=*/true) ? 1 : 0;
  }
  std::printf("\n%s: %d failing case(s)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails ? 1 : 0;
}
