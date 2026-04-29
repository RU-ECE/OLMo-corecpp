#pragma once
/**
 * include/olmo_cpp/model/feed_forward.hpp
 *
 * Declaration of the SwiGLU FeedForward sublayer used inside every
 * transformer block. SwiGLU forward:
 *
 *     gate = W1 x        (D -> H)
 *     up   = W3 x        (D -> H, separate matrix)
 *     y    = silu(gate) ⊙ up        (elementwise gated activation)
 *     out  = W2 y        (H -> D)
 *
 * silu(z) = z · sigmoid(z). See kernels/silu_mul.cu for a full
 * pedagogical writeup.
 *
 * The "fused" path concatenates W1 and W3 into a single 2H-wide
 * weight matrix so there's a single matmul launch instead of two —
 * ~constant launch-overhead win.
 *
 * --- Includes from this project ---
 *   (none — torch only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/feed_forward.cpp : implementation.
 *   - src/model/block.cpp / fused_block.cpp / block_variants.cpp :
 *     instantiates a FeedForward as one of the two sublayers in
 *     every transformer block.
 *
 * --- Role in training pipeline ---
 *   Foundational. Together with attention, this is most of the FLOPs
 *   in every transformer forward pass.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// SwiGLU feed-forward: out = w2(silu(w1(x)) * w3(x))
/// When use_fused_gate_up is true, w1 and w3 are combined into a single
/// [2*H, D] weight matrix for a single GEMM (halves launch overhead).
class FeedForwardImpl : public torch::nn::Module {
 public:
  FeedForwardImpl(int64_t d_model, int64_t hidden_size, bool bias = false,
                  bool use_fused_gate_up = false);

  torch::Tensor forward(torch::Tensor x);

 private:
  // Standard path: separate gate (w1) and up (w3)
  torch::nn::Linear w1_{nullptr};
  torch::nn::Linear w3_{nullptr};

  // Fused path: combined gate+up weight [2*H, D]
  torch::nn::Linear w_gate_up_{nullptr};

  // Down projection (always separate)
  torch::nn::Linear w2_{nullptr};

  bool fused_ = false;
};

TORCH_MODULE(FeedForward);

}  // namespace olmo_cpp
