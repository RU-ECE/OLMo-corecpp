#include "olmo_cpp/backend/fused_ops.hpp"
#include "olmo_cpp/backend/backend.hpp"
#include <cmath>

namespace olmo_cpp {
namespace fused {

// ---------------------------------------------------------------------------
// Fused QKV projection
// ---------------------------------------------------------------------------

QKVResult fused_qkv_projection(torch::Tensor x,
                                torch::Tensor weight_qkv,
                                std::optional<torch::Tensor> bias_qkv) {
  // weight_qkv is [3*D_out, D]. Single GEMM: x @ weight_qkv^T = [B, S, 3*D_out]
  auto qkv = torch::nn::functional::linear(x, weight_qkv,
      bias_qkv ? *bias_qkv : torch::Tensor());

  int64_t d_out = weight_qkv.size(0) / 3;
  auto chunks = qkv.split(d_out, /*dim=*/-1);

  return {chunks[0], chunks[1], chunks[2]};
}

// ---------------------------------------------------------------------------
// Fused attention
// ---------------------------------------------------------------------------

torch::Tensor fused_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                               std::optional<torch::Tensor> mask,
                               double scale, bool causal) {
  int64_t d = q.size(-1);
  if (scale <= 0) {
    scale = 1.0 / std::sqrt(static_cast<double>(d));
  }

  // Scaled dot-product attention
  auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;

  if (causal && !mask.has_value()) {
    int64_t sq = q.size(-2);
    int64_t sk = k.size(-2);
    auto causal_mask = torch::ones({sq, sk}, scores.options()).tril(sk - sq);
    scores = scores.masked_fill(causal_mask == 0, -1e9f);
  } else if (mask.has_value()) {
    scores = scores + *mask;
  }

  auto attn = torch::softmax(scores, -1);
  return torch::matmul(attn, v);
}

// ---------------------------------------------------------------------------
// Fused SwiGLU FFN
// ---------------------------------------------------------------------------

torch::Tensor fused_swiglu_ffn(torch::Tensor x,
                                torch::Tensor w_gate,
                                torch::Tensor w_up,
                                torch::Tensor w_down,
                                std::optional<torch::Tensor> bias_gate,
                                std::optional<torch::Tensor> bias_up,
                                std::optional<torch::Tensor> bias_down) {
  // Fused gate+up: concatenate weights, single GEMM, then split
  auto w_gate_up = torch::cat({w_gate, w_up}, /*dim=*/0);  // [2*H, D]
  std::optional<torch::Tensor> b_gate_up;
  if (bias_gate && bias_up) {
    b_gate_up = torch::cat({*bias_gate, *bias_up}, /*dim=*/0);
  }

  auto gate_up = torch::nn::functional::linear(x, w_gate_up,
      b_gate_up ? *b_gate_up : torch::Tensor());

  int64_t h = w_gate.size(0);
  auto gate = gate_up.narrow(-1, 0, h);
  auto up = gate_up.narrow(-1, h, h);

  // Fused SiLU * mul via backend
  auto hidden = get_backend().silu_mul(gate, up);

  // Down projection
  return torch::nn::functional::linear(hidden, w_down,
      bias_down ? *bias_down : torch::Tensor());
}

// ---------------------------------------------------------------------------
// Fused residual + norm
// ---------------------------------------------------------------------------

ResidualNormResult fused_residual_norm(torch::Tensor x,
                                        torch::Tensor residual,
                                        torch::Tensor norm_weight,
                                        double eps,
                                        double dropout_p,
                                        bool training) {
  ResidualNormResult result;

  // Apply dropout if in training mode
  torch::Tensor x_drop = x;
  if (training && dropout_p > 0) {
    x_drop = torch::dropout(x, dropout_p, /*train=*/true);
  }

  // Fused residual add + norm
  result.residual_out = x_drop + residual;
  result.normed = get_backend().rms_norm(result.residual_out, norm_weight, eps);

  return result;
}

// ---------------------------------------------------------------------------
// Fused gate+up projection
// ---------------------------------------------------------------------------

GateUpResult fused_gate_up_projection(torch::Tensor x,
                                       torch::Tensor weight_gate_up,
                                       std::optional<torch::Tensor> bias) {
  auto out = torch::nn::functional::linear(x, weight_gate_up,
      bias ? *bias : torch::Tensor());

  int64_t h = weight_gate_up.size(0) / 2;
  return {out.narrow(-1, 0, h), out.narrow(-1, h, h)};
}

}  // namespace fused
}  // namespace olmo_cpp
