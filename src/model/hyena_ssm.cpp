#include "olmo_cpp/model/hyena_ssm.hpp"
#include <torch/torch.h>
#include <cmath>

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// HyenaFilter
// ---------------------------------------------------------------------------

HyenaFilterImpl::HyenaFilterImpl(int64_t d_model, int64_t order,
                                   int64_t filter_order)
    : d_model_(d_model),
      order_(order),
      filter_order_(filter_order),
      w1_(register_module(
          "w1",
          torch::nn::Linear(
              torch::nn::LinearOptions(filter_order, filter_order).bias(true)))),
      w2_(register_module(
          "w2",
          torch::nn::Linear(
              torch::nn::LinearOptions(filter_order, order * d_model)
                  .bias(false))))
{
  // Learnable log-frequencies; initialised to log-space from 1 to sqrt(10000).
  // Using log parameterisation keeps frequencies positive after exp().
  const int64_t half = filter_order / 2;
  auto log_min = std::log(1.0);
  auto log_max = std::log(std::sqrt(10000.0));
  log_freqs_ = register_parameter(
      "log_freqs",
      torch::linspace(log_min, log_max, half));
}

torch::Tensor HyenaFilterImpl::forward(int64_t seq_len,
                                        torch::Device dev,
                                        torch::Dtype dt) {
  const int64_t half = filter_order_ / 2;

  // Normalised time positions in [0, 1): shape [S]
  auto t = torch::arange(seq_len, torch::TensorOptions().device(dev).dtype(dt))
               / static_cast<double>(seq_len); // [S]

  // Frequencies: [half] (positive, via exp of learnable log-freqs)
  auto freqs = torch::exp(log_freqs_.to(dev).to(dt)); // [half]

  // Positional encoding: [S, half] outer products, then sin + cos → [S, filter_order]
  auto angle = 2.0 * M_PI * t.unsqueeze(1) * freqs.unsqueeze(0); // [S, half]
  auto pos_emb = torch::cat({torch::sin(angle), torch::cos(angle)}, /*dim=*/1); // [S, filter_order]

  // MLP: [S, filter_order] → [S, order * d_model]
  auto h = torch::gelu(w1_(pos_emb));
  auto raw = w2_(h); // [S, order * d_model]

  // Reshape to [order, d_model, S] (the canonical conv filter layout)
  return raw.view({seq_len, order_, d_model_}).permute({1, 2, 0}).contiguous();
}

// ---------------------------------------------------------------------------
// HyenaSSM
// ---------------------------------------------------------------------------

HyenaSSMImpl::HyenaSSMImpl(const TransformerConfig& cfg, int64_t /*layer_idx*/)
    : order_(cfg.hyena_order),
      d_model_(cfg.d_model),
      out_proj_(register_module(
          "out_proj",
          torch::nn::Linear(
              torch::nn::LinearOptions(cfg.d_model, cfg.d_model).bias(false)))),
      filter_gen_(register_module(
          "filter_gen",
          HyenaFilter(cfg.d_model, cfg.hyena_order, cfg.hyena_filter_order)))
{
  // (order + 1) input projections: index 0 = value, 1..N = gates u_1..u_N
  for (int64_t i = 0; i <= order_; ++i) {
    auto lin = torch::nn::Linear(
        torch::nn::LinearOptions(cfg.d_model, cfg.d_model).bias(false));
    in_projs_->push_back(register_module("in_proj_" + std::to_string(i), lin));
  }
}

// Causal linear convolution via real FFT.
//   x: [B, D, S]   k: [D, S]  →  [B, D, S]
torch::Tensor HyenaSSMImpl::fft_conv(torch::Tensor x, torch::Tensor k) {
  const int64_t S = x.size(-1);
  int64_t n = 1;
  while (n < 2 * S) n <<= 1;

  auto xf = torch::fft::rfft(x, n, /*dim=*/-1); // [B, D, n/2+1]
  auto kf = torch::fft::rfft(k, n, /*dim=*/-1); // [D,    n/2+1]

  // Broadcast: [B, D, n/2+1] * [1, D, n/2+1]
  auto yf = xf * kf.unsqueeze(0);

  auto y = torch::fft::irfft(yf, n, /*dim=*/-1); // [B, D, n]
  return y.slice(/*dim=*/-1, 0, S);               // [B, D, S]
}

torch::Tensor HyenaSSMImpl::forward(torch::Tensor x) {
  const int64_t B = x.size(0);
  const int64_t S = x.size(1);
  const auto dev  = x.device();
  const auto dt   = x.scalar_type();

  // --- Input projections: (order+1) streams, each [B, D, S] ---
  // Lay them out as a list; index 0 is the value, 1..N are gates.
  std::vector<torch::Tensor> streams;
  streams.reserve(order_ + 1);
  for (int64_t i = 0; i <= order_; ++i) {
    auto lin = in_projs_->ptr<torch::nn::LinearImpl>(i);
    // lin->forward(x): [B, S, D] → permute to [B, D, S] for conv
    streams.push_back(lin->forward(x).permute({0, 2, 1}).contiguous());
  }

  // --- Generate all N filters: [order, D, S] ---
  auto filters = filter_gen_->forward(S, dev, dt); // [N, D, S]

  // --- Hyena recurrence ---
  // h_0 = v (streams[0])
  // h_i = conv(K_i, h_{i-1}) * u_i   for i = 1..N
  auto h = streams[0]; // [B, D, S]
  for (int64_t i = 0; i < order_; ++i) {
    auto ki = filters[i]; // [D, S]
    h = fft_conv(h, ki) * streams[i + 1]; // element-wise gate
  }

  // --- Output projection ---
  // h: [B, D, S] → [B, S, D] → project → [B, S, D]
  h = h.permute({0, 2, 1}).contiguous(); // [B, S, D]
  return out_proj_(h);                   // [B, S, d_model]
}

}  // namespace olmo_cpp
