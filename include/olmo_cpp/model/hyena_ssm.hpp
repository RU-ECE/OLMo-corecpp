#pragma once

/**
 * include/olmo_cpp/model/hyena_ssm.hpp
 *
 * Hyena SSM operator — a data-controlled, long-convolution sequence mixer.
 *
 * Hyena (Poli et al. 2023) replaces attention with an order-N composition of
 * implicit-filter convolutions gated by input projections:
 *
 *   Notation: * = causal FFT convolution, ⊙ = element-wise product
 *
 *   Project x → [v, u_1, u_2, ..., u_N]  (N+1 projections, each [B,D,S])
 *   Generate N filters K_1,...,K_N via positional MLP  [D, S] each
 *
 *   h_0 = v
 *   h_i = (K_i * h_{i-1}) ⊙ u_i      for i = 1..N
 *   out  = W_out · h_N
 *
 * Filter generation (implicit neural representation):
 *   t    = linspace(0, 1, S)
 *   freq = exp(linspace(log 1, log max_freq, emb_dim/2))
 *   pos  = [sin(2π·freq·t), cos(2π·freq·t)]   [S, emb_dim]
 *   K    = W2(GELU(W1(pos)))                    [S, N·D]
 *   K    = reshape → [N, D, S]
 *
 * Each K_i is thus a fixed (per-forward) filter, content-independent but
 * position-dependent. The u_i gates provide the content-dependence.
 * All convolutions are O(n log n) via real FFT.
 *
 * --- Callers ---
 *   src/model/wave_block.cpp : WaveBlockImpl owns one HyenaSSM when
 *   cfg.sequence_mixer == Hyena (or HybridHyena on even layers).
 */

#include "olmo_cpp/config.hpp"
#include <torch/torch.h>

namespace olmo_cpp {

// Generates N Hyena convolution filters from a positional MLP.
// A single HyenaFilter instance is shared across the N orders — the output
// projection produces N*D values that are split into N filters of size D.
class HyenaFilterImpl : public torch::nn::Module {
 public:
  /// d_model: channel width; order: N (number of filters); filter_order: MLP
  /// input embedding dimension (number of frequency features × 2).
  HyenaFilterImpl(int64_t d_model, int64_t order, int64_t filter_order);

  /// Returns filters of shape [order, d_model, seq_len] on (device, dtype).
  torch::Tensor forward(int64_t seq_len, torch::Device dev, torch::Dtype dt);

 private:
  int64_t d_model_;
  int64_t order_;
  int64_t filter_order_; // embedding dim = filter_order (must be even)

  torch::nn::Linear w1_; // filter_order → filter_order  (hidden layer)
  torch::nn::Linear w2_; // filter_order → order * d_model (output)

  // Learnable log-spaced frequencies [filter_order/2]
  torch::Tensor log_freqs_;
};

TORCH_MODULE(HyenaFilter);

// Full Hyena SSM operator.
class HyenaSSMImpl : public torch::nn::Module {
 public:
  HyenaSSMImpl(const TransformerConfig& cfg, int64_t layer_idx);

  /// x: [B, S, d_model] → [B, S, d_model]
  torch::Tensor forward(torch::Tensor x);

 private:
  int64_t order_;
  int64_t d_model_;

  // (order+1) input projections: first is the "value", rest are gates u_i
  torch::nn::ModuleList in_projs_; // each: d_model → d_model, no bias
  torch::nn::Linear out_proj_;     // d_model → d_model

  HyenaFilter filter_gen_;

  // Causal linear convolution: x [B,D,S], k [D,S] → [B,D,S]
  static torch::Tensor fft_conv(torch::Tensor x, torch::Tensor k);
};

TORCH_MODULE(HyenaSSM);

}  // namespace olmo_cpp
