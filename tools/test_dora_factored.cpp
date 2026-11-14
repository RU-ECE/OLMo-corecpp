/**
 * tools/test_dora_factored.cpp — factored DoRA == materialised DoRA, and how much faster.
 *
 *   1. float64: outputs and the gradients of x, lora_A, lora_B and magnitude
 *      agree between DoRAAdapter::forward_factored and forward_materialized.
 *   2. The check can fail: dropping the cross term 2 s B.(W A^T) from the
 *      norm must be caught.
 *   3. Timing (fp32 on the first available GPU, else CPU): forward + backward
 *      at a 7B FFN shape (4096 -> 11008, rank 32), both forms.
 *
 *   ./build/test_dora_factored [--tokens 2048] [--out 11008] [--in 4096] [--rank 32] [--device mps|cuda|cpu]
 */
#include "olmo_cpp/model/dora.hpp"

#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace olmo_cpp;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}
double rel(const torch::Tensor& a, const torch::Tensor& b) {
  return (a - b).abs().max().item<double>() / (b.abs().max().item<double>() + 1e-300);
}
void sync(const torch::Device& d) {
  if (d.is_mps()) torch::mps::synchronize();
  if (d.is_cuda()) torch::cuda::synchronize();
}
}  // namespace

int main(int argc, char** argv) {
  int64_t T = 2048, out = 11008, in = 4096, r = 32;
  std::string dev_s = torch::cuda::is_available() ? "cuda" : (torch::mps::is_available() ? "mps" : "cpu");
  for (int i = 1; i + 1 < argc; ++i) {
    std::string a = argv[i];
    if (a == "--tokens") T = std::stoll(argv[++i]);
    else if (a == "--out") out = std::stoll(argv[++i]);
    else if (a == "--in") in = std::stoll(argv[++i]);
    else if (a == "--rank") r = std::stoll(argv[++i]);
    else if (a == "--device") dev_s = argv[++i];
  }

  // 1 + 2: exactness in float64 on CPU.
  {
    torch::manual_seed(0);
    auto o = torch::TensorOptions().dtype(torch::kFloat64);
    DoRAAdapter ad(48, 40, 6, 12.0);
    ad->to(torch::kFloat64);
    ad->train();
    auto W = torch::randn({48, 40}, o);
    auto x = torch::randn({3, 7, 40}, o).requires_grad_(true);
    {
      torch::NoGradGuard ng;
      ad->forward_factored(x, W);  // lazy magnitude init (train mode)
      for (auto& p : ad->named_parameters()) {
        if (p.key() == "lora_B") p.value().normal_(0.0, 0.3);  // B != 0: every term of the norm is live
        if (p.key() == "magnitude") p.value().mul_(torch::rand_like(p.value()) + 0.5);
      }
    }
    std::vector<torch::Tensor> leaves = {x};
    for (auto& p : ad->parameters()) leaves.push_back(p);
    auto yF = ad->forward_factored(x, W);
    auto gF = torch::autograd::grad({(yF * yF).sum()}, leaves);
    auto yM = ad->forward_materialized(x, W);
    auto gM = torch::autograd::grad({(yM * yM).sum()}, leaves);
    double worst = 0.0;
    for (size_t i = 0; i < gF.size(); ++i) worst = std::max(worst, rel(gF[i], gM[i]));
    char b[200];
    std::snprintf(b, sizeof b, "factored == materialised (fp64): output rel err %.1e, worst grad rel err %.1e",
                  rel(yF, yM), worst);
    // Both forms compute the norm in fp32 by design (bf16 safety), so even in
    // fp64 they agree to fp32 precision; the planted error below is ~0.07.
    check(rel(yF, yM) < 1e-6 && worst < 1e-6, b);

    // Planted error: norm without the cross term 2 s B.(W A^T).
    auto named = ad->named_parameters();
    auto A = named["lora_A"], Bm = named["lora_B"], m = named["magnitude"];
    const double s = 12.0 / 6.0;
    auto rn_bad = (W.pow(2).sum(1) + s * s * (torch::matmul(Bm, torch::matmul(A, A.t())) * Bm).sum(1)).sqrt();
    auto y_bad = (torch::matmul(x, W.t()) + torch::matmul(torch::matmul(x, A.t()), Bm.t()) * s) * (m / rn_bad);
    std::snprintf(b, sizeof b, "planted error (no cross term in the norm) is caught (output rel err %.2f)",
                  rel(y_bad, yM));
    check(rel(y_bad, yM) > 1e-3, b);
  }

  // 3: speed at a real shape.
  torch::Device dev = dev_s == "cuda" ? torch::Device(torch::kCUDA)
                    : dev_s == "mps"  ? torch::Device(torch::kMPS)
                                      : torch::Device(torch::kCPU);
  for (auto dt : {torch::kFloat32, torch::kBFloat16}) {
    if (dt == torch::kBFloat16 && !dev.is_cuda()) continue;  // bf16 timing only where it is fast
    torch::manual_seed(1);
    auto o = torch::TensorOptions().dtype(dt).device(dev);
    DoRAAdapter ad(out, in, static_cast<int>(r), static_cast<double>(r));
    ad->to(dev, dt);
    ad->train();
    auto W = torch::randn({out, in}, o) * 0.02;
    auto x = torch::randn({T, in}, o).requires_grad_(true);
    {
      torch::NoGradGuard ng;
      ad->forward_factored(x, W);
      for (auto& p : ad->named_parameters())
        if (p.key() == "lora_B") p.value().normal_(0.0, 0.01);
    }
    auto time = [&](bool factored) {
      auto run = [&] {
        for (auto& p : ad->parameters()) p.mutable_grad() = torch::Tensor();
        x.mutable_grad() = torch::Tensor();
        auto y = factored ? ad->forward_factored(x, W) : ad->forward_materialized(x, W);
        y.sum().backward();
      };
      for (int i = 0; i < 2; ++i) run();
      sync(dev);
      std::vector<double> t;
      for (int i = 0; i < 6; ++i) {
        auto a = std::chrono::steady_clock::now();
        run();
        sync(dev);
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
      }
      std::sort(t.begin(), t.end());
      return t[t.size() / 2];
    };
    const double tm = time(false), tf = time(true);
    std::printf("timing %s %s, %lld tokens, %lld -> %lld, rank %lld: materialised %.1f ms, factored %.1f ms (%.2fx)\n",
                dev_s.c_str(), dt == torch::kFloat32 ? "fp32" : "bf16", (long long)T, (long long)in,
                (long long)out, (long long)r, tm, tf, tm / tf);
  }
  std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
  return g_fail ? 1 : 0;
}
