/**
 * src/backend/fused_lm_head_ce_autograd.cpp
 *
 * Autograd Function wrapping fused_lm_head_ce (item A3).
 *
 * Forward: call fused_lm_head_ce (CUDA kernel or CPU ref) to get the
 * scalar mean loss. Save (h, weight, labels, ignore_index) for backward.
 *
 * Backward through:
 *     loss = mean_{n: labels[n] != ignore_index} (
 *               logsumexp(h[n] @ W^T) - (h[n] @ W^T)[labels[n]] )
 *
 *     d loss / d logits[n, v] = (softmax(logits[n])[v] - 1[v == labels[n]])
 *                                  / valid_count           for non-ignored n
 *                            = 0                            otherwise
 *
 *     d loss / d h    = (softmax - onehot) @ W
 *     d loss / d W    = (softmax - onehot).T @ h
 *
 * We recompute logits = h @ W.T via fast_linear, derive softmax,
 * subtract one_hot at the label positions (and zero out ignored rows),
 * scale by grad_loss / valid_count, then do two fast_matmul calls.
 * Costs one extra GEMM vs the unfused backward, saves the
 * [B*S, V] logits + log_softmax materialization in forward — wash on
 * compute, big win on HBM at large V.
 */

#include "olmo_cpp/backend/fused_lm_head_ce.hpp"
#include "olmo_cpp/backend/cublas_direct.hpp"

#include <torch/torch.h>
#include <torch/csrc/autograd/custom_function.h>

namespace olmo_cpp {

namespace {

struct FusedLMHeadCEFunction
    : public torch::autograd::Function<FusedLMHeadCEFunction> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor h,
                                 torch::Tensor weight,
                                 torch::Tensor labels,
                                 int64_t ignore_index) {
    auto loss = fused_lm_head_ce(h, weight, labels, ignore_index);
    ctx->save_for_backward({h, weight, labels});
    ctx->saved_data["ignore_index"] = ignore_index;
    return loss;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto h      = saved[0];
    auto W      = saved[1];
    auto labels = saved[2];
    const int64_t ignore_index = ctx->saved_data["ignore_index"].toInt();
    auto grad_loss = grad_outputs[0];

    const int64_t N = h.size(0);
    const int64_t d = h.size(1);
    const int64_t V = W.size(0);

    // Recompute logits = h @ W.T via cuBLASLt-direct.
    auto logits = fast_linear(h, W, torch::Tensor());      // [N, V], dtype = h

    // softmax in compute-dtype. Numerically stable: subtract per-row max
    // before exp, divide by sum.
    auto max_per_row = std::get<0>(logits.max(/*dim=*/1, /*keepdim=*/true));
    auto exp_shift   = (logits - max_per_row).exp();
    auto softmax     = exp_shift / exp_shift.sum(/*dim=*/1, /*keepdim=*/true);

    // Build mask of valid rows: labels != ignore_index AND label in [0, V).
    auto valid_mask = (labels != ignore_index) &
                      (labels >= 0) & (labels < V);
    const int64_t valid_count_int =
        std::max<int64_t>(1, valid_mask.to(torch::kInt64).sum().item<int64_t>());
    const double inv_count =
        grad_loss.item<double>() / static_cast<double>(valid_count_int);

    // grad_logits = (softmax - onehot) * inv_count, zeroed where invalid.
    // Scatter -1 at (row, label) for valid rows. Use a safe label index
    // (clamp to 0 for invalid rows; their contribution is masked out).
    auto safe_labels = labels.clamp(0, V - 1).to(torch::kInt64);
    auto onehot = torch::zeros_like(softmax);
    onehot.scatter_(
        /*dim=*/1,
        safe_labels.unsqueeze(1),
        torch::ones_like(safe_labels.unsqueeze(1), softmax.options()));
    auto grad_logits = softmax - onehot;
    // Zero rows where labels are ignored (or out of range).
    grad_logits = grad_logits * valid_mask.to(softmax.dtype()).unsqueeze(1);
    grad_logits = grad_logits * static_cast<double>(inv_count);

    // grad_h = grad_logits @ W.  Shapes: [N, V] @ [V, d] = [N, d].
    auto grad_h = fast_matmul(grad_logits, W, /*transa=*/false, /*transb=*/false);
    // grad_W = grad_logits.T @ h. Shapes: [V, N] @ [N, d] = [V, d].
    auto grad_W = fast_matmul(grad_logits, h, /*transa=*/true,  /*transb=*/false);

    return {grad_h, grad_W, torch::Tensor(), torch::Tensor()};
  }
};

}  // namespace

torch::Tensor fused_lm_head_ce_autograd(torch::Tensor h,
                                          torch::Tensor weight,
                                          torch::Tensor labels,
                                          int64_t ignore_index) {
  return FusedLMHeadCEFunction::apply(h, weight, labels, ignore_index);
}

}  // namespace olmo_cpp
