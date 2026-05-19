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

torch::Tensor fused_ffn(torch::Tensor x,
                         torch::Tensor w_gate_up,
                         torch::Tensor w_down) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda()) {
    return fused_ffn_cuda(x, w_gate_up, w_down);
  }
#endif
  return fused_ffn_cpu(x, w_gate_up, w_down);
}

}  // namespace olmo_cpp
