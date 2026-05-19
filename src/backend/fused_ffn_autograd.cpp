/**
 * src/backend/fused_ffn_autograd.cpp
 *
 * Autograd Function wrapping fused_ffn (item 1 follow-on).
 *
 * Forward: call fused_ffn (CUDA kernel or CPU ref) and stash inputs
 * + the intermediate post-silu activation `act` (needed for backward).
 *
 * Backward through SwiGLU(gate_up, w_down) = w_down @ (silu(gate) * up):
 *   Let `gate, up = split(gate_up @ x)`. silu(g) = g·σ(g);
 *   d silu(g)/dg = σ(g) + g·σ(g)·(1-σ(g)) = σ(g) · (1 + g · (1 - σ(g))).
 *
 *   grad_act      = grad_y @ w_down               (no transpose: linear)
 *   grad_gate     = grad_act * up * d_silu(gate)
 *   grad_up       = grad_act * silu(gate)
 *   grad_gate_up  = cat([grad_gate, grad_up], dim=-1)
 *   grad_w_down   = grad_y.T @ act
 *   grad_x        = grad_gate_up @ w_gate_up
 *   grad_w_gate_up= grad_gate_up.T @ x
 *
 * All ops via ATen so autograd flows through the CUDA forward.
 */

#include "olmo_cpp/backend/fused_ffn.hpp"

#include <torch/torch.h>
#include <torch/csrc/autograd/custom_function.h>

namespace olmo_cpp {

namespace {

struct FusedFFNFunction : public torch::autograd::Function<FusedFFNFunction> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor x,
                                 torch::Tensor w_gate_up,
                                 torch::Tensor w_down) {
    // Compute the forward both ways: through ATen so we have the
    // intermediates for backward, and through the fused kernel for
    // the actual output. The fused-kernel value is the one returned;
    // intermediates are recomputed under the hood when needed in
    // backward (matches activation-checkpointing-style memory cost).
    auto y = fused_ffn(x, w_gate_up, w_down);
    ctx->save_for_backward({x, w_gate_up, w_down});
    return y;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto w_gate_up = saved[1];
    auto w_down = saved[2];
    auto grad_y = grad_outputs[0];

    const int64_t H = w_gate_up.size(0) / 2;
    auto gate_up = torch::nn::functional::linear(x, w_gate_up);  // [B,S,2H]
    auto gate = gate_up.narrow(-1, 0, H);
    auto up   = gate_up.narrow(-1, H, H);
    auto sig = torch::sigmoid(gate);
    auto silu_gate = gate * sig;                                  // silu(gate)
    auto act = silu_gate * up;                                    // [B,S,H]

    // grad_act = grad_y @ w_down  (y = act @ w_down.T)
    auto grad_act = torch::matmul(grad_y, w_down);                // [B,S,H]
    auto grad_w_down = torch::matmul(
        grad_y.flatten(0, -2).transpose(0, 1),                     // [d, B*S]
        act.flatten(0, -2));                                       // [B*S, H]

    // d silu(g)/dg = sig + g * sig * (1 - sig)
    auto d_silu_gate = sig + gate * sig * (1 - sig);
    auto grad_gate = grad_act * up * d_silu_gate;
    auto grad_up   = grad_act * silu_gate;
    auto grad_gate_up = torch::cat({grad_gate, grad_up}, /*dim=*/-1);  // [B,S,2H]

    auto grad_w_gate_up = torch::matmul(
        grad_gate_up.flatten(0, -2).transpose(0, 1),
        x.flatten(0, -2));
    auto grad_x = torch::matmul(grad_gate_up, w_gate_up);

    return {grad_x, grad_w_gate_up, grad_w_down};
  }
};

}  // namespace

torch::Tensor fused_ffn_autograd(torch::Tensor x,
                                   torch::Tensor w_gate_up,
                                   torch::Tensor w_down) {
  return FusedFFNFunction::apply(x, w_gate_up, w_down);
}

}  // namespace olmo_cpp
