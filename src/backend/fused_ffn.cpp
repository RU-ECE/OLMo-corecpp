/**
 * src/backend/fused_ffn.cpp
 *
 * CPU reference + host dispatcher for fused SwiGLU FFN (item I).
 */

#include "olmo_cpp/backend/fused_ffn.hpp"
#include "olmo_cpp/backend/backend.hpp"

#include <torch/torch.h>

namespace olmo_cpp {

torch::Tensor fused_ffn_cpu(torch::Tensor x,
                              torch::Tensor w_gate_up,
                              torch::Tensor w_down) {
  TORCH_CHECK(x.dim() == 3 && w_gate_up.dim() == 2 && w_down.dim() == 2,
              "fused_ffn_cpu: shapes wrong");
  // Reference path mirrors FeedForwardImpl::forward (fused-gate-up branch):
  //   gate_up = linear(x, w_gate_up)
  //   gate, up = split(gate_up, H, dim=-1)
  //   act = silu(gate) * up
  //   y = linear(act, w_down)
  auto gate_up = torch::nn::functional::linear(x, w_gate_up);   // [B,S,2H]
  const int64_t H = gate_up.size(-1) / 2;
  auto gate = gate_up.narrow(-1, 0, H);
  auto up   = gate_up.narrow(-1, H, H);
  auto act  = get_backend().silu_mul(gate, up);                 // [B,S,H]
  return torch::nn::functional::linear(act, w_down);             // [B,S,d]
}

// A1 — training-side dispatcher. Returns (y, gate_up). For CPU and
// non-aligned shapes, gate_up is computed via fast_linear so the
// numerics match the production forward path; CUDA-aligned shapes
// route through the WMMA/TMA train variants that produce gate_up
// as a kernel side-output (one extra HBM write per call).
std::pair<torch::Tensor, torch::Tensor>
fused_ffn_train(torch::Tensor x,
                 torch::Tensor w_gate_up,
                 torch::Tensor w_down) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda() && x.scalar_type() == torch::kBFloat16) {
    const int64_t d = x.size(-1);
    const int64_t H = w_gate_up.size(0) / 2;
    if (d % 16 == 0 && H % 16 == 0) {
      return fused_ffn_tma_train_cuda(x, w_gate_up, w_down);
    }
  }
#endif
  // CPU / non-aligned fallback: produce gate_up explicitly. Slightly
  // slower than the kernel-side write but exercised only on edge cases.
  auto gate_up = torch::nn::functional::linear(x, w_gate_up);
  auto y = fused_ffn(x, w_gate_up, w_down);
  return {y, gate_up};
}

torch::Tensor fused_ffn(torch::Tensor x,
                         torch::Tensor w_gate_up,
                         torch::Tensor w_down) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda() && x.scalar_type() == torch::kBFloat16) {
    const int64_t d = x.size(-1);
    const int64_t H = w_gate_up.size(0) / 2;
    if (d % 16 == 0 && H % 16 == 0) {
      // Tensor-core path — TMA variant on sm_90+, plain WMMA otherwise.
      // fused_ffn_tma_cuda itself runtime-checks and falls back to the
      // WMMA path on older arches, so we always dispatch here.
      return fused_ffn_tma_cuda(x, w_gate_up, w_down);
    }
    // Fallback to FMA-loop kernel for non-aligned shapes.
    return fused_ffn_cuda(x, w_gate_up, w_down);
  }
  if (x.is_cuda()) {
    return fused_ffn_cuda(x, w_gate_up, w_down);
  }
#endif
  return fused_ffn_cpu(x, w_gate_up, w_down);
}

}  // namespace olmo_cpp
