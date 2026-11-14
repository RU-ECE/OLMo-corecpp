/**
 * tools/test_slmtp.cpp — correctness of SL-MTP heads (model/sl_mtp_head.hpp).
 *
 *   1. Factored form == direct form, float64: the main loss and every head
 *      loss, and the gradients of h, W, g and every U_k, V_k, a_k.
 *   2. The gradient check can fail: dropping the path through rms(u_k) (the
 *      error the novelty review found in the first derivation) must be caught.
 *   3. At initialisation (U = 0, a = 1) every head's logits equal the main
 *      head's logits.
 *   4. Transformer integration: forward(labels) with mtp_head_type =
 *      shared_lowrank equals main + w * mean(heads) from the direct reference
 *      on the same backbone output, and drafting returns LMHead(u_k).
 * Exit 0 = all pass.
 */
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/sl_mtp_head.hpp"
#include "olmo_cpp/model/transformer.hpp"

#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>
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

struct Problem {
  torch::Tensor h, W, g, labels;
  std::vector<SLMTPHead> heads;
};

Problem make_problem(int64_t B, int64_t S, int64_t d, int64_t V, int64_t r, int64_t K) {
  torch::manual_seed(0);
  auto o = torch::TensorOptions().dtype(torch::kFloat64);
  Problem p;
  p.h = torch::randn({B, S, d}, o).requires_grad_(true);
  p.W = (torch::randn({V, d}, o) * 0.1).requires_grad_(true);
  p.g = (torch::rand({d}, o) + 0.5).requires_grad_(true);
  p.labels = torch::randint(0, V, {B, S}, torch::kInt64);
  p.labels.index_put_({0, 3}, -100);  // exercise ignore_index
  for (int64_t k = 0; k < K; ++k) {
    SLMTPHead hd(d, r);
    hd->to(torch::kFloat64);
    torch::NoGradGuard ng;
    hd->U.normal_(0.0, 0.3);  // non-zero so every gradient path is live
    hd->V.normal_(0.0, 0.3);
    hd->a.fill_(0.7 - 0.2 * static_cast<double>(k));
    p.heads.push_back(hd);
  }
  return p;
}

std::vector<torch::Tensor> leaves(Problem& p) {
  std::vector<torch::Tensor> v = {p.h, p.W, p.g};
  for (auto& hd : p.heads) {
    v.push_back(hd->U);
    v.push_back(hd->V);
    v.push_back(hd->a);
  }
  return v;
}

torch::Tensor total(const MTPLosses& L) {
  auto t = L.main;
  for (const auto& x : L.heads) t = t + 0.3 * x;
  return t;
}

std::vector<torch::Tensor> grads_of(Problem& p, const MTPLosses& L) {
  auto ls = leaves(p);
  return torch::autograd::grad({total(L)}, ls);
}

std::vector<const SLMTPHeadImpl*> raw(const Problem& p) {
  std::vector<const SLMTPHeadImpl*> v;
  for (const auto& hd : p.heads) v.push_back(hd.get());
  return v;
}

// The review's error: treat rms(u_k) as a constant in the backward.
MTPLosses factored_without_rms_grad(const Problem& p, double eps) {
  const int64_t S = p.h.size(1);
  MTPLosses out;
  auto ce = [](const torch::Tensor& lg, const torch::Tensor& lab) {
    return torch::nn::functional::cross_entropy(lg.reshape({-1, lg.size(-1)}), lab.reshape(-1),
                                                torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100));
  };
  auto y = torch::matmul(p.h * p.g, p.W.t());
  out.main = ce(y * torch::rsqrt(p.h.pow(2).mean(-1, true) + eps), p.labels);
  for (size_t k = 0; k < p.heads.size(); ++k) {
    const int64_t s = static_cast<int64_t>(k) + 1;
    const auto& hd = p.heads[k];
    auto hk = p.h.narrow(1, 0, S - s);
    auto c = torch::matmul(hk, hd->V);
    auto u = hd->a * hk + torch::matmul(c, hd->U.t());
    auto T = torch::matmul(p.W, p.g.unsqueeze(1) * hd->U);
    auto inv = torch::rsqrt(u.pow(2).mean(-1, true) + eps).detach();  // PLANTED ERROR
    out.heads.push_back(ce((hd->a * y.narrow(1, 0, S - s) + torch::matmul(c, T.t())) * inv,
                           p.labels.narrow(1, s, S - s)));
  }
  return out;
}

void test_factored_equals_direct() {
  const double eps = 1e-6;
  auto p = make_problem(2, 12, 48, 211, 6, 3);
  auto F = sl_mtp_losses(p.h, p.W, p.g, eps, raw(p), p.labels, -100);
  auto D = sl_mtp_losses_direct(p.h, p.W, p.g, eps, raw(p), p.labels, -100);
  double lerr = (F.main - D.main).abs().item<double>();
  for (size_t k = 0; k < F.heads.size(); ++k) lerr = std::max(lerr, (F.heads[k] - D.heads[k]).abs().item<double>());
  check(F.heads.size() == 3 && lerr < 1e-12, "factored losses == direct losses (max abs err " + std::to_string(lerr) + ")");

  auto gF = grads_of(p, F);
  auto gD = grads_of(p, D);
  const char* names[] = {"h", "W", "g"};
  double worst = 0.0;
  std::string worst_name;
  for (size_t i = 0; i < gF.size(); ++i) {
    const double e = rel(gF[i], gD[i]);
    if (e > worst) {
      worst = e;
      worst_name = i < 3 ? names[i] : ("head" + std::to_string((i - 3) / 3) + "." + "UVa"[(i - 3) % 3]);
    }
  }
  char buf[160];
  std::snprintf(buf, sizeof buf, "factored gradients == direct gradients for h, W, g, U, V, a (worst rel err %.1e at %s)",
                worst, worst_name.c_str());
  check(worst < 1e-10, buf);

  // 2. The same comparison must catch the missing rms(u) path.
  auto B = factored_without_rms_grad(p, eps);
  auto gB = grads_of(p, B);
  double bad = 0.0;
  for (size_t i = 3; i < gB.size(); ++i) bad = std::max(bad, rel(gB[i], gD[i]));
  std::snprintf(buf, sizeof buf, "planted error (no gradient through rms(u_k)) is caught (rel err %.2f)", bad);
  check(bad > 1e-3, buf);
}

void test_chunked() {
  const double eps = 1e-6, hw = 0.3;
  auto p = make_problem(2, 12, 48, 211, 6, 3);
  auto D = sl_mtp_losses_direct(p.h, p.W, p.g, eps, raw(p), p.labels, -100);
  auto ref = D.main + hw * (D.heads[0] + D.heads[1] + D.heads[2]) / 3.0;
  auto gD = torch::autograd::grad({ref}, leaves(p));
  std::vector<SLMTPHeadImpl*> hs;
  for (auto& hd : p.heads) hs.push_back(hd.get());
  for (int64_t chunk : {7, 24, 1000}) {  // 24 rows: a ragged last block, one block, one oversized block
    auto L = sl_mtp_loss_chunked(p.h, p.W, p.g, eps, hs, p.labels, -100, hw, chunk);
    auto gC = torch::autograd::grad({L * 1.7}, leaves(p));  // non-unit upstream gradient
    double worst = 0.0;
    for (size_t i = 0; i < gC.size(); ++i) worst = std::max(worst, rel(gC[i], gD[i] * 1.7));
    char b[200];
    std::snprintf(b, sizeof b, "chunked loss (block %lld rows) == direct: loss err %.1e, worst grad rel err %.1e",
                  (long long)chunk, (L - ref).abs().item<double>(), worst);
    check((L - ref).abs().item<double>() < 1e-12 && worst < 1e-10, b);
  }
  // Frozen W (the kill test's setting): no dW, everything else unchanged.
  p.W.set_requires_grad(false);
  auto L = sl_mtp_loss_chunked(p.h, p.W, p.g, eps, hs, p.labels, -100, hw, 5);
  std::vector<torch::Tensor> rest = {p.h, p.g};
  for (auto& hd : p.heads) { rest.push_back(hd->U); rest.push_back(hd->V); rest.push_back(hd->a); }
  auto gF = torch::autograd::grad({L}, rest);
  double worst = 0.0;
  for (size_t i = 0; i < gF.size(); ++i) worst = std::max(worst, rel(gF[i], gD[i == 0 ? 0 : i + 1]));
  char b[160];
  std::snprintf(b, sizeof b, "chunked loss with W frozen: gradients still exact (worst rel err %.1e)", worst);
  check(worst < 1e-10, b);
  p.W.set_requires_grad(true);
}

// CUDA: the row kernels (kernels/sl_mtp_ce.cu) must give the reference
// path's loss and gradients, in fp32 and bf16.
void test_kernel_cuda() {
  if (!torch::cuda::is_available()) return;
  for (auto dt : {torch::kFloat32, torch::kBFloat16}) {
    torch::manual_seed(4);
    auto o = torch::TensorOptions().dtype(dt).device(torch::kCUDA);
    const int64_t B = 2, S = 40, d = 64, V = 3001, r = 8, K = 3;
    auto h = torch::randn({B, S, d}, o).requires_grad_(true);
    auto W = (torch::randn({V, d}, o) * 0.1).requires_grad_(true);
    auto g = (torch::rand({d}, o) + 0.5).requires_grad_(true);
    auto labels = torch::randint(0, V, {B, S}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    std::vector<SLMTPHead> hs;
    std::vector<SLMTPHeadImpl*> raw;
    std::vector<torch::Tensor> leaves = {h, W, g};
    for (int64_t k = 0; k < K; ++k) {
      SLMTPHead hd(d, r);
      hd->to(torch::kCUDA, dt);
      torch::NoGradGuard ng;
      hd->U.normal_(0.0, 0.3);
      hd->V.normal_(0.0, 0.3);
      hd->a.fill_(0.8);
      hs.push_back(hd);
      raw.push_back(hd.get());
      leaves.push_back(hd->U);
      leaves.push_back(hd->V);
      leaves.push_back(hd->a);
    }
    auto run = [&](bool kernel) {
      if (kernel) unsetenv("OLMO_SLMTP_NO_KERNEL");
      else setenv("OLMO_SLMTP_NO_KERNEL", "1", 1);
      auto L = sl_mtp_loss_chunked(h, W, g, 1e-6, raw, labels, -100, 0.3, 32);
      auto gr = torch::autograd::grad({L}, leaves);
      unsetenv("OLMO_SLMTP_NO_KERNEL");
      return std::make_pair(L, gr);
    };
    auto [Lk, gk] = run(true);
    auto [Lr, gr] = run(false);
    double worst = 0.0;
    for (size_t i = 0; i < gk.size(); ++i) worst = std::max(worst, rel(gk[i].to(torch::kFloat64), gr[i].to(torch::kFloat64)));
    const double tol = dt == torch::kFloat32 ? 1e-4 : 3e-2;
    char b[200];
    std::snprintf(b, sizeof b, "CUDA row kernels == reference path (%s): loss %.6f vs %.6f, worst grad rel err %.1e",
                  dt == torch::kFloat32 ? "fp32" : "bf16", Lk.item<double>(), Lr.item<double>(), worst);
    check(std::abs(Lk.item<double>() - Lr.item<double>()) < tol * std::abs(Lr.item<double>()) && worst < tol, b);
  }
}

void test_init_is_main_head() {
  auto o = torch::TensorOptions().dtype(torch::kFloat64);
  torch::manual_seed(1);
  auto h = torch::randn({1, 5, 32}, o), W = torch::randn({97, 32}, o), g = torch::rand({32}, o) + 0.5;
  SLMTPHead hd(32, 4);
  hd->to(torch::kFloat64);
  auto lm = [&](const torch::Tensor& x) {
    return torch::matmul(x * torch::rsqrt(x.pow(2).mean(-1, true) + 1e-6) * g, W.t());
  };
  const double e = (lm(hd->mix(h)) - lm(h)).abs().max().item<double>();
  check(e < 1e-12, "at init (U = 0, a = 1) every head's logits equal the main logits (err " + std::to_string(e) + ")");
}

void test_transformer() {
  TransformerConfig c;
  c.d_model = 64;
  c.n_layers = 2;
  c.n_heads = 4;
  c.vocab_size = 151;
  c.hidden_size_multiple_of = 32;
  c.num_mtp_heads = 3;
  c.mtp_head_type = "shared_lowrank";
  c.mtp_rank = 8;
  c.mtp_loss_weight = 0.25;
  torch::manual_seed(2);
  Transformer m(c);
  m->init_weights();
  m->to(torch::kFloat64);
  std::vector<const SLMTPHeadImpl*> heads;
  int64_t n_sl = 0;
  for (auto& it : m->named_parameters())
    if (it.key().rfind("mtp_heads.", 0) == 0) {
      ++n_sl;
      torch::NoGradGuard ng;
      it.value().normal_(0.0, 0.2);  // move off the init point so the heads differ
    }
  check(n_sl == 3 * c.num_mtp_heads, "3 SL-MTP heads register U, V, a (" + std::to_string(n_sl) + " tensors)");

  auto ids = torch::randint(0, c.vocab_size, {2, 10}, torch::kInt64);
  auto labels = torch::randint(0, c.vocab_size, {2, 10}, torch::kInt64);
  auto loss = m->forward(ids, labels, -100);

  // Reference from the backbone output and the direct form.
  auto h = m->forward_backbone(ids);
  auto params = m->named_parameters();
  std::vector<SLMTPHead> copies;
  for (int64_t k = 0; k < c.num_mtp_heads; ++k) {
    SLMTPHead hd(c.d_model, c.mtp_rank);
    hd->to(torch::kFloat64);
    torch::NoGradGuard ng;
    const std::string pre = "mtp_heads." + std::to_string(k) + ".";
    hd->U.copy_(params[pre + "U"]);
    hd->V.copy_(params[pre + "V"]);
    hd->a.copy_(params[pre + "a"]);
    copies.push_back(hd);
  }
  for (const auto& hd : copies) heads.push_back(hd.get());
  auto D = sl_mtp_losses_direct(h, params["lm_head.w_out.weight"], params["lm_head.norm.weight"],
                                c.layer_norm_eps, heads, labels, -100);
  auto ref = D.main + c.mtp_loss_weight * (D.heads[0] + D.heads[1] + D.heads[2]) / 3.0;
  const double e = (loss - ref).abs().item<double>();
  check(e < 1e-10, "Transformer::forward(labels) == main + w * mean(heads) from the direct form (err " +
                       std::to_string(e) + ")");
  loss.backward();
  int64_t with_grad = 0, total = 0;
  for (auto& it : m->named_parameters()) {
    ++total;
    if (it.value().grad().defined() && it.value().grad().abs().sum().item<double>() > 0) ++with_grad;
  }
  check(with_grad == total, "every parameter gets a gradient (" + std::to_string(with_grad) + "/" +
                                std::to_string(total) + ")");

  torch::NoGradGuard ng;
  auto hl = h.select(1, 9);  // [2, d] -> take row 0
  auto drafts = m->forward_mtp_draft(hl.select(0, 0));
  double de = 0.0;
  for (int64_t k = 0; k < c.num_mtp_heads; ++k) {
    auto u = heads[static_cast<size_t>(k)]->mix(hl.select(0, 0).view({1, 1, -1}));
    auto want = m->apply_lm_head(u).reshape({-1});
    de = std::max(de, (drafts[static_cast<size_t>(k)].reshape({-1}) - want).abs().max().item<double>());
  }
  check(drafts.size() == 3 && de < 1e-10, "forward_mtp_draft returns LMHead(u_k) for each head (err " +
                                              std::to_string(de) + ")");
  // Drafting from the main logits (no W read) must give the same logits.
  auto h0 = hl.select(0, 0);
  auto main = m->apply_lm_head(h0.view({1, 1, -1})).reshape({-1});
  auto fast = m->forward_mtp_draft(h0, main);
  double fe = 0.0;
  for (size_t k = 0; k < fast.size(); ++k)
    fe = std::max(fe, (fast[k].reshape({-1}) - drafts[k].reshape({-1})).abs().max().item<double>());
  char b[128];
  std::snprintf(b, sizeof b, "drafting from the main logits (no W read) == direct drafting (err %.1e)", fe);
  // The model's LMHead norm runs in fp32 by design (IBackend::rms_norm), so
  // the direct path carries fp32 rounding; the factored path is exact fp64.
  check(fast.size() == 3 && fe < 1e-5, b);
}

}  // namespace

int main() {
  test_factored_equals_direct();
  test_chunked();
  test_kernel_cuda();
  test_init_is_main_head();
  test_transformer();
  std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
  return g_fail ? 1 : 0;
}
