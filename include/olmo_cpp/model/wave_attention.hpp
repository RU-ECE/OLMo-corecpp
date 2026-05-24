#pragma once

/**
 * include/olmo_cpp/model/wave_attention.hpp
 *
 * Content-modulated wave-field attention.
 *
 * Replaces the O(n²) QK dot-product with a learnable damped-wave convolution
 * kernel applied via FFT in O(n log n):
 *
 *   w_h(t) = exp(-α_h · t) · cos(ω_h · t + φ_h),  t = 0,1,…,S-1  (causal)
 *
 * Three scalar parameters per head (α, ω, φ) replace the full Q/K weight
 * matrices. Content-dependence is recovered through:
 *   1. A learned value projection  v  = W_v · x
 *   2. A learned sigmoid gate      g  = σ(W_gate · x)
 *   3. Output:  y = g ⊙ FFT_conv(w, v),  then W_out · y
 *
 * The gate makes the effective kernel input-dependent while keeping the
 * convolution itself O(n log n). Gradient flows through the FFT via
 * torch::fft::rfft / irfft (both are differentiable in LibTorch ≥ 1.13).
 *
 * --- Callers ---
 *   src/model/wave_block.cpp : WaveBlockImpl owns one WaveAttention
 */

#include "olmo_cpp/config.hpp"
#include <torch/torch.h>

namespace olmo_cpp {

class WaveAttentionImpl : public torch::nn::Module {
 public:
  WaveAttentionImpl(const TransformerConfig& cfg, int64_t layer_idx);

  /// x: [B, S, d_model] → [B, S, d_model]
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Linear w_v_;    // d_model → n_heads * head_dim
  torch::nn::Linear w_gate_; // d_model → n_heads * head_dim  (content gate)
  torch::nn::Linear w_out_;  // n_heads * head_dim → d_model

  // Per-head wave kernel parameters (raw; α passed through softplus in forward)
  torch::Tensor alpha_raw_; // [n_heads]
  torch::Tensor omega_;     // [n_heads]
  torch::Tensor phi_;       // [n_heads]

  int64_t n_heads_;
  int64_t head_dim_;

  // Build causal wave kernel [n_heads, S] on the given device/dtype.
  torch::Tensor build_kernel(int64_t S, torch::Device dev, torch::Dtype dt);

  // Causal linear convolution via FFT.
  // kern: [H, 1, S]   vals: [B, H, Dh, S]  → [B, H, Dh, S]
  static torch::Tensor fft_conv(torch::Tensor kern, torch::Tensor vals);
};

TORCH_MODULE(WaveAttention);

}  // namespace olmo_cpp
