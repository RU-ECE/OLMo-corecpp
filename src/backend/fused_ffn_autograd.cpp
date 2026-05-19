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
#include "olmo_cpp/backend/cublas_direct.hpp"

#include <torch/torch.h>
#include <torch/csrc/autograd/custom_function.h>

namespace olmo_cpp {

namespace {

struct FusedFFNFunction : public torch::autograd::Function<FusedFFNFunction> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor x,
                                 torch::Tensor w_gate_up,
                                 torch::Tensor w_down) {
    // A1 — call the training-side fused path that also publishes
    // gate_up. Saving it lets backward skip the recompute matmul.
    // gate_up is dropped if the caller never invokes backward; the
    // training-only kernel write happens regardless when this fwd
    // function fires, which is exactly the no-grad-disabled case.
    auto [y, gate_up] = fused_ffn_train(x, w_gate_up, w_down);
    ctx->save_for_backward({x, w_gate_up, w_down, gate_up});
    return y;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x         = saved[0];
    auto w_gate_up = saved[1];
    auto w_down    = saved[2];
    auto gate_up   = saved[3];   // A1 — saved from forward; no recompute
    auto grad_y    = grad_outputs[0];

    const int64_t H = w_gate_up.size(0) / 2;
    const int64_t d = x.size(-1);
    auto leading = x.sizes().vec();
    leading.pop_back();
    auto twoH_shape = leading;
    twoH_shape.push_back(2 * H);

    // Derive elementwise intermediates from the saved gate_up. The
    // gate/up narrows are views (free). sigmoid, silu_gate, act are
    // each one bandwidth-bound elementwise pass over [B*S*H] — cheap
    // vs the 103-GFLOP gate_up recompute they replace.
    auto gate      = gate_up.narrow(-1, 0, H);
    auto up        = gate_up.narrow(-1, H, H);
    auto sig       = torch::sigmoid(gate);
    auto silu_gate = gate * sig;
    auto act       = silu_gate * up;

    // grad_act = grad_y @ w_down  (cuBLASLt 2-D direct call).
    auto grad_y_flat = grad_y.reshape({-1, d});
    auto grad_act_flat = fast_matmul(grad_y_flat, w_down, false, false);

    // grad_w_down = grad_y.T @ act.
    auto act_flat = act.reshape({-1, H});
    auto grad_w_down = fast_matmul(grad_y_flat, act_flat, true, false);

    // d silu(g)/dg = sig + g * sig * (1 - sig).
    auto d_silu_gate = sig + gate * sig * (1 - sig);

    // Allocate grad_gate_up once and write the two halves directly into
    // it — no torch::cat, which would allocate a third tensor and copy
    // both halves into it. Mul_out targets a pre-allocated buffer.
    auto grad_gate_up = torch::empty(twoH_shape, grad_y.options());
    auto grad_gate_view = grad_gate_up.narrow(-1, 0, H);
    auto grad_up_view   = grad_gate_up.narrow(-1, H, H);
    auto grad_act = grad_act_flat.view(act.sizes());
    torch::mul_out(grad_gate_view, grad_act, up);
    grad_gate_view.mul_(d_silu_gate);
    torch::mul_out(grad_up_view, grad_act, silu_gate);

    // grad_w_gate_up = grad_gate_up.T @ x.
    auto grad_gate_up_flat = grad_gate_up.reshape({-1, 2 * H});
    auto x_flat = x.reshape({-1, d});
    auto grad_w_gate_up = fast_matmul(grad_gate_up_flat, x_flat, true, false);

    // grad_x = grad_gate_up @ w_gate_up.
    auto grad_x_flat = fast_matmul(grad_gate_up_flat, w_gate_up, false, false);
    auto grad_x = grad_x_flat.view(x.sizes());

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
