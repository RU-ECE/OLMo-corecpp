/**
 * tools/bench_slmtp.cpp — measured cost of MTP heads: full vs SL-MTP.
 *
 * Times forward + backward of the vocabulary-side work (everything after the
 * backbone output h) for one micro-batch, with the same plain-autograd cross
 * entropy in every variant so only the head math differs:
 *   main      next-token head only (the baseline every variant pays)
 *   full      + K heads as llm-cpp builds them today: proj (d x d), RMSNorm,
 *             then the shared LM head (its own d x V GEMM per head)
 *   sl-direct + K SL-MTP heads computed directly (LMHead(u_k), d x V each)
 *   sl-fact   + K SL-MTP heads in factored form (one shared d x V product)
 * Reports the median over iterations and the marginal cost per head.
 *
 *   ./build/bench_slmtp [--device mps|cpu|cuda] [--d 2048] [--vocab 50304]
 *       [--tokens 1024] [--heads 3] [--rank 64] [--iters 10] [--bf16] [--chunk 512]
 */
#include "olmo_cpp/backend/fused_lm_head_ce.hpp"
#include "olmo_cpp/model/sl_mtp_head.hpp"

#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace olmo_cpp;

namespace {

void sync(const torch::Device& d) {
  if (d.is_mps()) torch::mps::synchronize();
  if (d.is_cuda()) torch::cuda::synchronize();
}

double median_ms(const std::function<void()>& f, int iters, const torch::Device& dev) {
  for (int i = 0; i < 2; ++i) f();  // warmup
  sync(dev);
  std::vector<double> t;
  for (int i = 0; i < iters; ++i) {
    auto a = std::chrono::steady_clock::now();
    f();
    sync(dev);
    t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

torch::Tensor ce(const torch::Tensor& lg, const torch::Tensor& lab) {
  return torch::nn::functional::cross_entropy(lg.reshape({-1, lg.size(-1)}), lab.reshape(-1));
}

}  // namespace

int main(int argc, char** argv) {
  std::string dev_s = torch::mps::is_available() ? "mps" : "cpu";
  int64_t d = 2048, Vocab = 50304, N = 1024, K = 3, r = 64;
  int iters = 10;
  int64_t chunk = 512;
  bool bf16 = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nx = [&] { return std::string(argv[++i]); };
    if (a == "--device") dev_s = nx();
    else if (a == "--d") d = std::stoll(nx());
    else if (a == "--vocab") Vocab = std::stoll(nx());
    else if (a == "--tokens") N = std::stoll(nx());
    else if (a == "--heads") K = std::stoll(nx());
    else if (a == "--rank") r = std::stoll(nx());
    else if (a == "--iters") iters = std::stoi(nx());
    else if (a == "--bf16") bf16 = true;
    else if (a == "--chunk") chunk = std::stoll(nx());
  }
  torch::Device dev = dev_s == "cuda" ? torch::Device(torch::kCUDA)
                    : dev_s == "mps"  ? torch::Device(torch::kMPS)
                                      : torch::Device(torch::kCPU);
  auto dt = bf16 ? torch::kBFloat16 : torch::kFloat32;
  auto o = torch::TensorOptions().dtype(dt).device(dev);
  const double eps = 1e-6;
  torch::manual_seed(0);
  auto h = torch::randn({1, N, d}, o).requires_grad_(true);
  auto W = (torch::randn({Vocab, d}, o) * 0.02).requires_grad_(true);
  auto g = torch::ones({d}, o).requires_grad_(true);
  auto labels = torch::randint(0, Vocab, {1, N}, torch::TensorOptions().dtype(torch::kInt64).device(dev));

  // Full heads as built today: proj (d x d) + per-head RMSNorm, then LMHead.
  std::vector<torch::Tensor> P, Gk;
  for (int64_t k = 0; k < K; ++k) {
    P.push_back((torch::randn({d, d}, o) * 0.02).requires_grad_(true));
    Gk.push_back(torch::ones({d}, o).requires_grad_(true));
  }
  std::vector<SLMTPHead> sl;
  std::vector<const SLMTPHeadImpl*> slp;
  for (int64_t k = 0; k < K; ++k) {
    SLMTPHead hd(d, r);
    hd->to(dev, dt);
    torch::NoGradGuard ng;
    hd->U.normal_(0.0, 0.02);
    sl.push_back(hd);
    slp.push_back(hd.get());
  }
  auto rmsn = [&](const torch::Tensor& x, const torch::Tensor& w) {
    return x * torch::rsqrt(x.pow(2).mean(-1, true) + eps) * w;
  };
  auto lm = [&](const torch::Tensor& x) { return torch::matmul(rmsn(x, g), W.t()); };

  auto run_main = [&] { ce(lm(h), labels).backward(); };
  auto run_full = [&] {
    auto loss = ce(lm(h), labels);
    for (int64_t k = 0; k < K; ++k) {
      const int64_t s = k + 1;
      auto u = rmsn(torch::matmul(h.narrow(1, 0, N - s), P[static_cast<size_t>(k)].t()), Gk[static_cast<size_t>(k)]);
      loss = loss + ce(lm(u), labels.narrow(1, s, N - s));
    }
    loss.backward();
  };
  auto run_sl = [&](bool factored) {
    auto L = factored ? sl_mtp_losses(h, W, g, eps, slp, labels, -100)
                      : sl_mtp_losses_direct(h, W, g, eps, slp, labels, -100);
    auto loss = L.main;
    for (auto& x : L.heads) loss = loss + x;
    loss.backward();
  };
  // Production paths: full heads through the repo's fused LM-head CE (what
  // Transformer::forward runs for mtp_head_type=full), and SL-MTP through the
  // chunked loss (what it runs for shared_lowrank). Both weight heads as 1/K.
  auto lab_flat = labels.reshape(-1);
  auto run_full_fused = [&] {
    auto loss = fused_lm_head_ce_autograd(rmsn(h, g).reshape({-1, d}), W, lab_flat, -100);
    torch::Tensor hs;
    for (int64_t k = 0; k < K; ++k) {
      const int64_t s = k + 1;
      auto u = rmsn(torch::matmul(h.narrow(1, 0, N - s), P[static_cast<size_t>(k)].t()), Gk[static_cast<size_t>(k)]);
      auto l = fused_lm_head_ce_autograd(rmsn(u, g).reshape({-1, d}), W, labels.narrow(1, s, N - s).reshape(-1), -100);
      hs = hs.defined() ? hs + l : l;
    }
    (loss + hs / static_cast<double>(K)).backward();
  };
  std::vector<SLMTPHeadImpl*> slm;
  for (auto& hd : sl) slm.push_back(hd.get());
  auto run_sl_chunk = [&] { sl_mtp_loss_chunked(h, W, g, eps, slm, labels, -100, 1.0, chunk).backward(); };
  auto zero = [&] {
    for (auto* t : {&h, &W, &g}) t->mutable_grad() = torch::Tensor();
    for (auto& t : P) t.mutable_grad() = torch::Tensor();
    for (auto& t : Gk) t.mutable_grad() = torch::Tensor();
    for (auto& hd : sl)
      for (auto& p : hd->parameters()) p.mutable_grad() = torch::Tensor();
  };

  std::printf("device %s  dtype %s  d=%lld  V=%lld  tokens=%lld  heads=%lld  rank=%lld  iters=%d\n",
              dev_s.c_str(), bf16 ? "bf16" : "fp32", (long long)d, (long long)Vocab, (long long)N,
              (long long)K, (long long)r, iters);
  const double t_main = median_ms([&] { zero(); run_main(); }, iters, dev);
  const double t_full = median_ms([&] { zero(); run_full(); }, iters, dev);
  const double t_dir = median_ms([&] { zero(); run_sl(false); }, iters, dev);
  const double t_fac = median_ms([&] { zero(); run_sl(true); }, iters, dev);
  const double t_ffu = median_ms([&] { zero(); run_full_fused(); }, iters, dev);
  const double t_chk = median_ms([&] { zero(); run_sl_chunk(); }, iters, dev);

  // Analytic forward FLOPs per token for the vocabulary side (backward ~2x).
  const double main_f = 2.0 * d * Vocab;
  const double full_f = main_f + K * (2.0 * d * d + 2.0 * d * Vocab);
  const double dir_f = main_f + K * (4.0 * d * r + 2.0 * d * Vocab);
  const double fac_f = main_f + K * (4.0 * d * r + 2.0 * Vocab * r + 2.0 * d * Vocab * r / N);
  std::printf("%-10s %10s %14s %16s %16s\n", "variant", "fwd+bwd ms", "per head ms", "fwd GF/token", "vs full (time)");
  auto row = [&](const char* n, double t, double f) {
    std::printf("%-10s %10.1f %14.1f %16.3f %15.0f%%\n", n, t, K > 0 ? (t - t_main) / K : 0.0, f / 1e9,
                100.0 * (t - t_full) / t_full);
  };
  // Decode-time drafting, one position: today's path stacks the K head inputs
  // and runs them through the LM head (one GEMM that reads all of W); the SL-MTP
  // path reuses the main logits the verify pass already produced (reads K x T).
  {
    torch::NoGradGuard ng;
    auto h1 = h.detach().select(1, 0).select(0, 0);  // [d]
    auto main_logits = lm(h1.view({1, d})).reshape({-1});
    std::vector<SLMTPHeadImpl*> mp;
    for (auto& hd : sl) mp.push_back(hd.get());
    auto Wd = W.detach(), gd = g.detach();
    const double t_old = median_ms([&] {
      std::vector<torch::Tensor> us;
      for (auto* hd : mp) us.push_back(hd->mix(h1.view({1, d})));
      auto stacked = torch::cat(us, 0);  // [K, d]
      auto out = torch::matmul(rmsn(stacked, gd), Wd.t());
      (void)out;
    }, iters * 5, dev);
    const double t_new = median_ms([&] { auto out = sl_mtp_draft_logits(h1, main_logits, Wd, gd, eps, mp); }, iters * 5, dev);
    std::printf("draft, one position, %lld heads: LM-head GEMM %.3f ms, from main logits %.3f ms (%.1fx); bytes read %.1f MB vs %.1f MB\n",
                (long long)K, t_old, t_new, t_old / t_new, Vocab * d * (bf16 ? 2.0 : 4.0) / 1e6,
                K * Vocab * r * (bf16 ? 2.0 : 4.0) / 1e6);
  }
  row("main", t_main, main_f);
  row("full", t_full, full_f);
  row("sl-direct", t_dir, dir_f);
  row("sl-fact", t_fac, fac_f);
  std::printf("production paths (no [tokens x vocab] tensor kept):\n");
  row("full-fused", t_ffu, full_f);
  row("sl-chunk", t_chk, fac_f);
  std::printf("sl-chunk vs full-fused: %+.0f%%\n", 100.0 * (t_chk - t_ffu) / t_ffu);
  return 0;
}
