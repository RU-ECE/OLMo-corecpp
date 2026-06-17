/**
 * src/backend/fused_lm_head_ce_autograd.cpp
 *
 * Autograd Function for the LM-head cross-entropy loss.
 *
 * Forward: fast_linear (cuBLASLt GEMM, tensor cores) → standard PyTorch CE.
 *   The original "A3 fused" approach used a scalar GEMV kernel that reads W
 *   [V, d] non-coalesced once per output row (32768 reads × 103 MB = 1.8 TB
 *   of non-coalesced HBM traffic at ~100 GB/s ≈ 18 s per head).
 *   cuBLASLt reads W once, reuses it across all N rows via the L2 cache, and
 *   uses tensor cores → ~2 ms for the GEMM + ~1 ms for CE per head.
 *   The 3.3 GB logit tensor is allocated but freed immediately by CE; the
 *   HBM cost of writing and reading it back is 3.3 GB / 3.35 TB/s ≈ 1 ms.
 *
 * Backward: recompute logits = fast_linear(h, W), derive softmax, subtract
 *   one_hot, scale by grad / valid_count, then two GEMMs for grad_h, grad_W.
 */

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
    // cuBLASLt GEMM (tensor cores) + standard CE: ~2ms per head vs ~18s
    // for the scalar GEMV in fused_lm_head_ce (non-coalesced W reads).
    auto logits = fast_linear(h, weight, torch::Tensor());   // [N, V]
    namespace F = torch::nn::functional;
    auto loss = F::cross_entropy(
        logits, labels,
        F::CrossEntropyFuncOptions().ignore_index(ignore_index));
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
    auto valid_f = valid_mask.to(softmax.dtype());            // [N], 0/1 in compute dtype
    // Keep count AND scale as DEVICE scalars — no .item() D2H sync in the
    // backward (this ran every step, once per MTP head, stalling the pipeline).
    // Count/divide in fp32 (bf16 can't represent counts > 256 exactly), then
    // cast the scalar back to compute dtype so grad_logits keeps its dtype.
    auto valid_count = valid_mask.to(torch::kFloat32).sum().clamp_min(1.0);   // [.]
    auto scale = (grad_loss.to(torch::kFloat32) / valid_count)
                     .to(softmax.dtype());                                     // [.]

    // grad_logits = (softmax - onehot) * scale, zeroed where invalid.
    // Scatter 1 at (row, label) for valid rows. Use a safe label index
    // (clamp to 0 for invalid rows; their contribution is masked out).
    auto safe_labels = labels.clamp(0, V - 1).to(torch::kInt64);
    auto onehot = torch::zeros_like(softmax);
    onehot.scatter_(
        /*dim=*/1,
        safe_labels.unsqueeze(1),
        torch::ones_like(safe_labels.unsqueeze(1), softmax.options()));
    auto grad_logits = (softmax - onehot) * valid_f.unsqueeze(1) * scale;

    // grad_h = grad_logits @ W.  Shapes: [N, V] @ [V, d] = [N, d].
    auto grad_h = torch::matmul(grad_logits, W);
    auto grad_W = torch::matmul(grad_logits.transpose(0, 1), h);

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
