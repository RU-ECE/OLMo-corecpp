/**
 * src/model/sl_mtp_head.cpp — SL-MTP heads; the math is in sl_mtp_head.hpp.
 */
#include "olmo_cpp/model/sl_mtp_head.hpp"

#include <cstdlib>

#ifdef OLMO_HAS_CUDA_KERNELS
namespace olmo_cpp {  // kernels/sl_mtp_ce.cu
void sl_mtp_ce_rowstats_cuda(const torch::Tensor& y, const torch::Tensor& p, const torch::Tensor& a,
                             const torch::Tensor& s, const torch::Tensor& labels, torch::Tensor& lse,
                             torch::Tensor& picked);
void sl_mtp_ce_grad_cuda(const torch::Tensor& y, const torch::Tensor& p, const torch::Tensor& a,
                         const torch::Tensor& coef, const torch::Tensor& s, const torch::Tensor& lse,
                         const torch::Tensor& labels, const torch::Tensor& w, torch::Tensor* gs,
                         torch::Tensor& dy, torch::Tensor& gz, torch::Tensor& gsy);
}  // namespace olmo_cpp
#endif

namespace olmo_cpp {

SLMTPHeadImpl::SLMTPHeadImpl(int64_t d_model, int64_t rank) {
  TORCH_CHECK(rank > 0 && rank <= d_model, "SL-MTP rank must be in [1, d_model], got ", rank);
  U = register_parameter("U", torch::zeros({d_model, rank}));
  V = register_parameter("V", torch::zeros({d_model, rank}));
  a = register_parameter("a", torch::ones({}));
  reset(0.02);
}

void SLMTPHeadImpl::reset(double std) {
  torch::NoGradGuard ng;
  U.zero_();
  a.fill_(1.0);
  V.normal_(0.0, std).clamp_(-3 * std, 3 * std);
}

const torch::Tensor& SLMTPHeadImpl::draft_T(const torch::Tensor& W, const torch::Tensor& g) {
  auto key_of = [](const torch::Tensor& t) {
    return std::vector<int64_t>{reinterpret_cast<int64_t>(t.data_ptr()), static_cast<int64_t>(t._version())};
  };
  std::vector<int64_t> key = key_of(W);
  for (const torch::Tensor* t : {&g, static_cast<const torch::Tensor*>(&U)}) {
    auto k = key_of(*t);
    key.insert(key.end(), k.begin(), k.end());
  }
  key.push_back(static_cast<int64_t>(W.scalar_type()));
  if (!T_.defined() || key != T_key_) {
    torch::NoGradGuard ng;
    T_ = torch::matmul(W, g.unsqueeze(1).to(W.scalar_type()) * U.to(W.scalar_type()));  // [V, r]
    T_key_ = std::move(key);
  }
  return T_;
}

std::vector<torch::Tensor> sl_mtp_draft_logits(const torch::Tensor& h, const torch::Tensor& main_logits,
                                               const torch::Tensor& W, const torch::Tensor& g, double eps,
                                               const std::vector<SLMTPHeadImpl*>& heads) {
  torch::NoGradGuard ng;
  auto hf = h.to(main_logits.scalar_type());
  // y = W (g ⊙ h) recovered from the main logits: LMHead(h) = y * rsqrt(ms(h) + eps).
  auto y = main_logits * torch::sqrt(hf.pow(2).mean(-1, true) + eps);
  std::vector<torch::Tensor> out;
  out.reserve(heads.size());
  for (auto* hd : heads) {
    auto V = hd->V.to(hf.scalar_type()), U = hd->U.to(hf.scalar_type()), a = hd->a.to(hf.scalar_type());
    auto c = torch::matmul(hf, V);
    auto u = a * hf + torch::matmul(c, U.t());
    const auto& T = hd->draft_T(W, g).to(hf.scalar_type());
    out.push_back((a * y + torch::matmul(c, T.t())) * torch::rsqrt(u.pow(2).mean(-1, true) + eps));
  }
  return out;
}

torch::Tensor SLMTPHeadImpl::mix(const torch::Tensor& h) const {
  return a * h + torch::matmul(torch::matmul(h, V), U.t());
}

namespace {

torch::Tensor ce(const torch::Tensor& logits, const torch::Tensor& labels, int64_t ignore_index) {
  const int64_t Vocab = logits.size(-1);
  return torch::nn::functional::cross_entropy(
      logits.reshape({-1, Vocab}), labels.reshape(-1),
      torch::nn::functional::CrossEntropyFuncOptions().ignore_index(ignore_index));
}

torch::Tensor inv_rms(const torch::Tensor& x, double eps) {
  return torch::rsqrt(x.pow(2).mean(-1, /*keepdim=*/true) + eps);
}

}  // namespace

MTPLosses sl_mtp_losses(const torch::Tensor& h, const torch::Tensor& W, const torch::Tensor& g,
                        double eps, const std::vector<const SLMTPHeadImpl*>& heads,
                        const torch::Tensor& labels, int64_t ignore_index) {
  TORCH_CHECK(g.defined(), "SL-MTP needs the LM head's final norm gain");
  const int64_t S = h.size(1);
  MTPLosses out;
  // The one shared vocabulary product: y = W (g ⊙ h), [B, S, V].
  auto y = torch::matmul(h * g, W.t());
  out.main = ce(y * inv_rms(h, eps), labels, ignore_index);
  auto gcol = g.unsqueeze(1);  // [d, 1]: diag(g) folded into each T_k
  for (size_t k = 0; k < heads.size(); ++k) {
    const int64_t shift = static_cast<int64_t>(k) + 1;
    if (shift >= S) break;
    const auto* hd = heads[k];
    auto hk = h.narrow(1, 0, S - shift);
    auto c = torch::matmul(hk, hd->V);                       // [B, S', r]
    auto u = hd->a * hk + torch::matmul(c, hd->U.t());       // [B, S', d], O(d r) per token
    auto T = torch::matmul(W, gcol * hd->U);                 // [V, r] = W diag(g) U, once per step
    auto logits = (hd->a * y.narrow(1, 0, S - shift) + torch::matmul(c, T.t())) * inv_rms(u, eps);
    out.heads.push_back(ce(logits, labels.narrow(1, shift, S - shift), ignore_index));
  }
  return out;
}

MTPLosses sl_mtp_losses_direct(const torch::Tensor& h, const torch::Tensor& W, const torch::Tensor& g,
                               double eps, const std::vector<const SLMTPHeadImpl*>& heads,
                               const torch::Tensor& labels, int64_t ignore_index) {
  const int64_t S = h.size(1);
  MTPLosses out;
  auto lm_head = [&](const torch::Tensor& x) { return torch::matmul(x * inv_rms(x, eps) * g, W.t()); };
  out.main = ce(lm_head(h), labels, ignore_index);
  for (size_t k = 0; k < heads.size(); ++k) {
    const int64_t shift = static_cast<int64_t>(k) + 1;
    if (shift >= S) break;
    auto u = heads[k]->mix(h.narrow(1, 0, S - shift));
    out.heads.push_back(ce(lm_head(u), labels.narrow(1, shift, S - shift), ignore_index));
  }
  return out;
}

}  // namespace olmo_cpp

// ── Chunked loss with gradients formed in the forward ─────────────────────
namespace olmo_cpp {
namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

// Per-row CE over one block: returns {sum of losses, G = w * (softmax - onehot)}
// with G zero on ignored rows. logits: [c, V] in the compute dtype.
std::pair<torch::Tensor, torch::Tensor> ce_block(const torch::Tensor& logits, const torch::Tensor& lab,
                                          const torch::Tensor& valid, double w, bool want_grad) {
  auto lse = torch::logsumexp(logits, -1, /*keepdim=*/true);                    // [c, 1]
  auto safe = lab.clamp_min(0).unsqueeze(1);                                      // [c, 1]
  auto picked = logits.gather(1, safe);                                           // [c, 1]
  auto vf = valid.to(logits.scalar_type()).unsqueeze(1);
  auto loss = ((lse - picked) * vf).sum() * w;  // stays on device: no sync per block
  if (!want_grad) return {loss, torch::Tensor()};
  auto G = (logits - lse).exp_();                                                 // softmax
  G.scatter_add_(1, safe, -torch::ones_like(picked));
  G.mul_(vf * w);
  return {loss, G};
}

struct SLMTPChunkedCE : public torch::autograd::Function<SLMTPChunkedCE> {
  static torch::Tensor forward(AutogradContext* ctx, torch::Tensor h, torch::Tensor W, torch::Tensor g,
                               at::TensorList head_params, torch::Tensor labels, double eps,
                               int64_t ignore_index, double head_weight, int64_t chunk) {
    const int64_t B = h.size(0), S = h.size(1), d = h.size(2), K = static_cast<int64_t>(head_params.size()) / 3;
    const auto cdt = h.scalar_type() == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32;
    const auto io = h.scalar_type();
    const bool gh = h.requires_grad(), gW = W.requires_grad(), gg = g.requires_grad();
    bool ghead = false;
    for (const auto& t : head_params) ghead = ghead || t.requires_grad();
    const bool any = gh || gW || gg || ghead;

    auto hf = h.reshape({-1, d});
    const int64_t N = hf.size(0);
    // Per-head labels on the same N rows: head k at (b, t) targets labels[b, t + k + 1].
    std::vector<torch::Tensor> lab(K + 1), valid(K + 1);
    lab[0] = labels.reshape(-1);
    for (int64_t k = 1; k <= K; ++k) {
      auto l = torch::full({B, S}, ignore_index, labels.options());
      if (k < S) l.narrow(1, 0, S - k).copy_(labels.narrow(1, k, S - k));
      lab[k] = l.reshape(-1);
    }
    std::vector<double> w(K + 1);
    for (int64_t k = 0; k <= K; ++k) {
      valid[k] = lab[k] != ignore_index;
      const double n = std::max<double>(1.0, valid[k].sum().item<double>());
      w[k] = k == 0 ? 1.0 / n : head_weight / (static_cast<double>(K) * n);
    }

    std::vector<torch::Tensor> U(K), V(K), a(K), T(K), dU(K), dV(K), da(K), dT(K), Gk(K);
    std::vector<int64_t> off(K + 1, 0);
    auto gc = g.to(cdt);
    for (int64_t k = 0; k < K; ++k) {
      U[k] = head_params[3 * k];
      V[k] = head_params[3 * k + 1];
      a[k] = head_params[3 * k + 2];
      Gk[k] = (g.unsqueeze(1) * U[k]).to(W.scalar_type());               // diag(g) U_k, [d, r]
      off[k + 1] = off[k] + U[k].size(1);
    }
    // T_k = W diag(g) U_k for every head in ONE pass over W.
    auto Tcat = K > 0 ? torch::matmul(W, torch::cat(Gk, 1)) : torch::Tensor();  // [V, sum r]
    for (int64_t k = 0; k < K; ++k) {
      T[k] = Tcat.narrow(1, off[k], off[k + 1] - off[k]);
      if (any) {
        dU[k] = torch::zeros(U[k].sizes(), U[k].options().dtype(cdt));
        dV[k] = torch::zeros(V[k].sizes(), V[k].options().dtype(cdt));
        da[k] = torch::zeros({}, a[k].options().dtype(cdt));
        dT[k] = torch::zeros(T[k].sizes(), T[k].options().dtype(cdt));
      }
    }
#ifdef OLMO_HAS_CUDA_KERNELS
    // Row kernels (kernels/sl_mtp_ce.cu) replace the per-head elementwise chain:
    // they never materialise a head's logits. OLMO_SLMTP_NO_KERNEL=1 forces the
    // reference path below (tools/test_slmtp compares the two).
    const bool kern = h.is_cuda() && cdt == torch::kFloat32 && std::getenv("OLMO_SLMTP_NO_KERNEL") == nullptr;
#else
    const bool kern = false;
#endif
    // bf16/fp16 weights on the kernel path: accumulate dW with mixed-dtype GEMMs
    // (bf16 x bf16 -> fp32 in place), no [V, d] temporaries or fp32 copies of W.
    const bool mixed = kern && W.scalar_type() == h.scalar_type() &&
                       (W.scalar_type() == torch::kBFloat16 || W.scalar_type() == torch::kHalf);
    torch::Tensor dW = (any && gW) ? torch::zeros(W.sizes(), W.options().dtype(cdt)) : torch::Tensor();
    torch::Tensor dg = any ? torch::zeros({d}, g.options().dtype(cdt)) : torch::Tensor();
    torch::Tensor dh = any ? torch::empty({N, d}, h.options().dtype(cdt)) : torch::Tensor();

    auto loss = torch::zeros({}, h.options().dtype(cdt));
    auto f32 = h.options().dtype(torch::kFloat32);
    auto zero_f = torch::zeros({}, f32), one_f = torch::ones({}, f32);
    for (int64_t i0 = 0; i0 < N; i0 += chunk) {
      const int64_t c = std::min(chunk, N - i0);
      auto hc = hf.narrow(0, i0, c);
      auto hcc = hc.to(cdt);
      auto hg = hc * g;                                                   // [c, d] (io dtype)
      auto y_io = torch::matmul(hg, W.t()).contiguous();                  // [c, V], shared by all heads
      auto inv_h = torch::rsqrt(hcc.pow(2).mean(-1, true) + eps);         // [c, 1]
      torch::Tensor y, dy, d_inv_h, dhc;
      if (any) dhc = torch::zeros({c, d}, hcc.options());
#ifdef OLMO_HAS_CUDA_KERNELS
      if (kern) {
        auto lab0 = lab[0].narrow(0, i0, c).contiguous();
        auto w0 = (valid[0].narrow(0, i0, c).to(torch::kFloat32) * w[0]).contiguous();
        auto s0 = inv_h.reshape({c}).to(torch::kFloat32).contiguous();
        auto lse = torch::empty({c}, f32), picked = torch::empty({c}, f32);
        sl_mtp_ce_rowstats_cuda(y_io, y_io, zero_f, s0, lab0, lse, picked);  // z = 0*y + y
        loss += ((lse - picked) * w0).sum();
        auto gz = torch::empty({c}, f32), gsy = torch::empty({c}, f32);
        if (any) {
          dy = torch::zeros({c, y_io.size(1)}, f32);
          auto gz0 = torch::empty({c}, f32);  // its own buffer: gz is overwritten per head below
          sl_mtp_ce_grad_cuda(y_io, y_io, zero_f, one_f, s0, lse, lab0, w0, nullptr, dy, gz0, gsy);
          d_inv_h = gz0.unsqueeze(1);
        }
        for (int64_t k = 0; k < K; ++k) {
          auto Vk = V[k].to(cdt), Uk = U[k].to(cdt), ak = a[k].to(cdt);
          auto ck = torch::matmul(hcc, Vk);                               // [c, r]
          auto u = ak * hcc + torch::matmul(ck, Uk.t());                 // [c, d]
          auto inv_u = torch::rsqrt(u.pow(2).mean(-1, true) + eps);
          auto Tio = T[k].to(y_io.scalar_type());
          auto ck_io = ck.to(y_io.scalar_type());
          auto pk = torch::matmul(ck_io, Tio.t()).contiguous();          // c T^T, [c, V]
          auto af = a[k].to(torch::kFloat32).reshape({});
          auto sk = inv_u.reshape({c}).to(torch::kFloat32).contiguous();
          auto labk = lab[k + 1].narrow(0, i0, c).contiguous();
          auto wk = (valid[k + 1].narrow(0, i0, c).to(torch::kFloat32) * w[k + 1]).contiguous();
          sl_mtp_ce_rowstats_cuda(y_io, pk, af, sk, labk, lse, picked);
          loss += ((lse - picked) * wk).sum();
          if (!any) continue;
          auto gs = torch::empty_like(pk);
          sl_mtp_ce_grad_cuda(y_io, pk, af, af, sk, lse, labk, wk, &gs, dy, gz, gsy);
          da[k] += gsy.sum().to(cdt);
          auto dck = torch::matmul(gs, Tio).to(cdt);                     // [c, r]
          dT[k] += torch::matmul(gs.t(), ck_io).to(cdt);                 // [V, r]
          auto du = -gz.unsqueeze(1).to(cdt) * inv_u.pow(3) * u / static_cast<double>(d);
          dhc += du * ak;
          da[k] += (du * hcc).sum();
          dck += torch::matmul(du, Uk);
          dU[k] += torch::matmul(du.t(), ck);
          dhc += torch::matmul(dck, Vk.t());
          dV[k] += torch::matmul(hcc.t(), dck);
        }
      } else
#endif
      {
      y = y_io.to(cdt);
      auto main_logits = y * inv_h;
      auto [l0, G0] = ce_block(main_logits, lab[0].narrow(0, i0, c), valid[0].narrow(0, i0, c), w[0], any);
      loss += l0;
      if (any) {
        dy = G0 * inv_h;
        d_inv_h = (G0 * y).sum(-1, true);
      }
      for (int64_t k = 0; k < K; ++k) {
        auto Vk = V[k].to(cdt), Uk = U[k].to(cdt), ak = a[k].to(cdt), Tk = T[k].to(cdt);
        auto ck = torch::matmul(hcc, Vk);                                 // [c, r]
        auto u = ak * hcc + torch::matmul(ck, Uk.t());                   // [c, d]
        auto inv_u = torch::rsqrt(u.pow(2).mean(-1, true) + eps);
        auto z = torch::addmm(ak * y, ck, Tk.t());                       // a y + c T^T, [c, V]
        auto [lk, Gk] = ce_block(z * inv_u, lab[k + 1].narrow(0, i0, c), valid[k + 1].narrow(0, i0, c),
                                 w[k + 1], any);
        loss += lk;
        if (!any) continue;
        auto Gs = Gk * inv_u;                                             // dL/dz
        dy += Gs * ak;
        da[k] += (Gs * y).sum();
        auto dck = torch::matmul(Gs, Tk);                                 // [c, r]
        dT[k] += torch::matmul(Gs.t(), ck);                               // [V, r]
        auto d_inv_u = (Gk * z).sum(-1, true);
        auto du = -d_inv_u * inv_u.pow(3) * u / static_cast<double>(d);  // through rms(u)
        dhc += du * ak;
        da[k] += (du * hcc).sum();
        dck += torch::matmul(du, Uk);
        dU[k] += torch::matmul(du.t(), ck);
        dhc += torch::matmul(dck, Vk.t());
        dV[k] += torch::matmul(hcc.t(), dck);
      }
      }
      if (!any) continue;
      auto dyi = dy.to(W.scalar_type());
      auto dhg = torch::matmul(dyi, W).to(cdt);                           // ONE dh GEMM for all heads
      if (gW) {                                                           // ONE dW GEMM for all heads
        if (mixed) at::addmm_out(dW, dW, dyi.t(), hg, torch::kFloat32);
        else dW += torch::matmul(dyi.t(), hg).to(cdt);
      }
      dhc += dhg * gc;
      dg += (dhg * hcc).sum(0);
      dhc += -d_inv_h.to(cdt) * inv_h.pow(3) * hcc / static_cast<double>(d);
      dh.narrow(0, i0, c).copy_(dhc);
    }
    if (any && K > 0) {  // back through T_k = W diag(g) U_k
      if (mixed) {
        // All heads at once: W^T [dT_1 .. dT_K] and dW += [dT_1 .. dT_K] [G_1 .. G_K]^T,
        // two GEMMs that read W once, in bf16 with fp32 output.
        auto dTcat = torch::cat(dT, 1).to(W.scalar_type());              // [V, sum r]
        auto WtdT = at::mm(W.t(), dTcat, torch::kFloat32);               // [d, sum r]
        if (gW) at::addmm_out(dW, dW, dTcat, torch::cat(Gk, 1).t(), torch::kFloat32);
        for (int64_t k = 0; k < K; ++k) {
          auto Wt_dT = WtdT.narrow(1, off[k], off[k + 1] - off[k]);
          dg += (Wt_dT * U[k].to(cdt)).sum(1);
          dU[k] += gc.unsqueeze(1) * Wt_dT;
        }
      } else {
        for (int64_t k = 0; k < K; ++k) {
          auto Wt_dT = torch::matmul(W.t().to(cdt), dT[k]);               // [d, r]
          if (gW) dW += torch::matmul(dT[k], (gc.unsqueeze(1) * U[k].to(cdt)).t());
          dg += (Wt_dT * U[k].to(cdt)).sum(1);
          dU[k] += gc.unsqueeze(1) * Wt_dT;
        }
      }
    }
    if (any) {
      variable_list saved = {dh.view({B, S, d}).to(io),
                             gW ? dW.to(W.scalar_type()) : torch::zeros({0}, W.options()),  // 0-size = none
                             dg.to(g.scalar_type())};
      for (int64_t k = 0; k < K; ++k) {
        saved.push_back(dU[k].to(U[k].scalar_type()));
        saved.push_back(dV[k].to(V[k].scalar_type()));
        saved.push_back(da[k].to(a[k].scalar_type()));
      }
      ctx->saved_data["grads"] = saved;
    }
    ctx->saved_data["K"] = K;
    return loss;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_out) {
    const int64_t K = ctx->saved_data["K"].toInt();
    auto go = grad_out[0];
    variable_list out;
    auto grads = ctx->saved_data["grads"].toTensorList().vec();
    auto scale = [&](const torch::Tensor& t) {
      return (t.defined() && t.numel() > 0) ? t * go.to(t.scalar_type()) : torch::Tensor();
    };
    out.push_back(scale(grads[0]));  // h
    out.push_back(scale(grads[1]));  // W
    out.push_back(scale(grads[2]));  // g
    for (int64_t i = 0; i < 3 * K; ++i) out.push_back(scale(grads[3 + i]));
    for (int i = 0; i < 5; ++i) out.emplace_back();  // labels, eps, ignore_index, head_weight, chunk
    return out;
  }
};

}  // namespace

torch::Tensor sl_mtp_loss_chunked(const torch::Tensor& h, const torch::Tensor& W, const torch::Tensor& g,
                                  double eps, const std::vector<SLMTPHeadImpl*>& heads,
                                  const torch::Tensor& labels, int64_t ignore_index, double head_weight,
                                  int64_t chunk_rows) {
  TORCH_CHECK(g.defined(), "SL-MTP needs the LM head's final norm gain");
  std::vector<torch::Tensor> ps;
  for (auto* hd : heads) {
    ps.push_back(hd->U);
    ps.push_back(hd->V);
    ps.push_back(hd->a);
  }
  return SLMTPChunkedCE::apply(h, W, g, at::TensorList(ps), labels, eps, ignore_index, head_weight, chunk_rows);
}

}  // namespace olmo_cpp
