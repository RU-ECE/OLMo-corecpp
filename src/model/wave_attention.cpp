#include "olmo_cpp/model/wave_attention.hpp"
#include <torch/torch.h>
#include <cmath>

namespace olmo_cpp {

WaveAttentionImpl::WaveAttentionImpl(const TransformerConfig& cfg,
                                     int64_t /*layer_idx*/)
    : w_v_(register_module(
          "w_v",
          torch::nn::Linear(torch::nn::LinearOptions(cfg.d_model,
                                                      cfg.n_heads * cfg.get_head_dim())
                                .bias(false)))),
      w_gate_(register_module(
          "w_gate",
          torch::nn::Linear(torch::nn::LinearOptions(cfg.d_model,
                                                      cfg.n_heads * cfg.get_head_dim())
                                .bias(false)))),
      w_out_(register_module(
          "w_out",
          torch::nn::Linear(torch::nn::LinearOptions(cfg.n_heads * cfg.get_head_dim(),
                                                      cfg.d_model)
                                .bias(false)))),
      n_heads_(cfg.n_heads),
      head_dim_(cfg.get_head_dim())
{
  // Wave kernel parameters — one scalar per head.
  // alpha_raw is passed through softplus so alpha stays positive.
  // Initialise: alpha ~ 0.1 (mild decay), omega ~ 1 (one cycle over seq),
  // phi = 0 (cosine, symmetric at t=0).
  alpha_raw_ = register_parameter(
      "alpha_raw",
      torch::full({n_heads_}, std::log(std::exp(0.1) - 1.0)));  // softplus⁻¹(0.1)
  omega_ = register_parameter("omega", torch::ones({n_heads_}));
  phi_   = register_parameter("phi",   torch::zeros({n_heads_}));
}

// Build causal damped-wave kernel for sequence length S.
// Returns [n_heads, S] on (dev, dt).
torch::Tensor WaveAttentionImpl::build_kernel(int64_t S,
                                               torch::Device dev,
                                               torch::Dtype dt) {
  // t: [1, S] integer offsets 0..S-1
  auto t = torch::arange(S, torch::TensorOptions().device(dev).dtype(dt))
               .unsqueeze(0); // [1, S]

  // α through softplus ensures positivity without clamping the gradient.
  auto alpha = torch::nn::functional::softplus(alpha_raw_)
                   .to(dev).to(dt)
                   .unsqueeze(1); // [H, 1]
  auto omega = omega_.to(dev).to(dt).unsqueeze(1); // [H, 1]
  auto phi   = phi_.to(dev).to(dt).unsqueeze(1);   // [H, 1]

  // kernel[h, t] = exp(-α_h * t) * cos(ω_h * t + φ_h)
  auto kernel = torch::exp(-alpha * t) * torch::cos(omega * t + phi); // [H, S]
  return kernel;
}

// Causal linear convolution via real FFT.
//   kern: [H, 1, S]    vals: [B, H, Dh, S]  →  [B, H, Dh, S]
// Pads to the next power-of-2 >= 2*S to ensure linear (not circular) conv.
torch::Tensor WaveAttentionImpl::fft_conv(torch::Tensor kern,
                                           torch::Tensor vals) {
  const int64_t S = vals.size(-1);
  int64_t n = 1;
  while (n < 2 * S) n <<= 1; // next power-of-two

  // rfft along last dimension; both tensors are real → complex output
  auto vf = torch::fft::rfft(vals, n, /*dim=*/-1); // [B, H, Dh, n/2+1]
  auto kf = torch::fft::rfft(kern, n, /*dim=*/-1); // [H,  1,  n/2+1]

  // Broadcasting: [B,H,Dh,n/2+1] * [H,1,n/2+1] → [B,H,Dh,n/2+1]
  auto yf = vf * kf;

  auto y = torch::fft::irfft(yf, n, /*dim=*/-1); // [B, H, Dh, n]
  return y.slice(/*dim=*/-1, 0, S);               // [B, H, Dh, S]
}

torch::Tensor WaveAttentionImpl::forward(torch::Tensor x) {
  const int64_t B  = x.size(0);
  const int64_t S  = x.size(1);
  const int64_t H  = n_heads_;
  const int64_t Dh = head_dim_;
  const auto dev   = x.device();
  const auto dt    = x.scalar_type();

  // --- Value and gate projections ---
  // [B, S, H*Dh]
  auto v    = w_v_(x);
  auto gate = torch::sigmoid(w_gate_(x));

  // Reshape to [B, H, Dh, S] for per-head processing
  v    = v.view({B, S, H, Dh}).permute({0, 2, 3, 1}).contiguous();    // [B,H,Dh,S]
  gate = gate.view({B, S, H, Dh}).permute({0, 2, 3, 1}).contiguous(); // [B,H,Dh,S]

  // --- Wave kernel: [H, S] → reshape to [H, 1, S] for broadcasting ---
  auto kern = build_kernel(S, dev, dt).unsqueeze(1); // [H, 1, S]

  // --- Causal FFT convolution ---
  auto y = fft_conv(kern, v); // [B, H, Dh, S]

  // --- Content-modulation gate ---
  y = gate * y; // [B, H, Dh, S]

  // Reshape back: [B, H, Dh, S] → [B, S, H*Dh]
  y = y.permute({0, 3, 1, 2}).contiguous().view({B, S, H * Dh});

  return w_out_(y); // [B, S, d_model]
}

}  // namespace olmo_cpp
