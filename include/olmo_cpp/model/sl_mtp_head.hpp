#pragma once
/**
 * include/olmo_cpp/model/sl_mtp_head.hpp
 *
 * SL-MTP: shared-logit low-rank multi-token-prediction heads.
 *
 * Head k (predicting token t+k+2 from the backbone output h at t) is
 *     u_k      = a_k h + U_k V_k^T h            (U_k, V_k: d x r; a_k: scalar)
 *     logits_k = LMHead(u_k) = W (g ⊙ u_k) * rsqrt(mean(u_k^2) + eps)
 * with the model's own final-norm gain g and unembedding W. Because LMHead is
 * linear up to the per-row scale, this factors exactly:
 *     logits_k = (a_k y + T_k c_k) * rsqrt(mean(u_k^2) + eps)
 *     y   = W (g ⊙ h)       [V]   the main head's product, shared by every head
 *     c_k = V_k^T h         [r]
 *     T_k = W diag(g) U_k   [V x r], built once per step, not per token
 * So a head's marginal cost is ~2·V·r FLOPs per token (T_k c_k) instead of
 * the 2·d·V of its own vocabulary projection, and in the backward every head's
 * gradient flows through the single shared y, so autograd issues one W^T
 * GEMM and one dW GEMM for the main head and all K heads together.
 *
 * Initialisation: U_k = 0, a_k = 1, V_k random, so every head starts as the
 * main head. a_k is learnable: if training drives it to 0 the "residual on
 * the main logits" prior was wrong and the head is a plain rank-r head.
 *
 * Prior art this sits next to (cite it): Zhang et al. 2507.15130 (frozen
 * unembedding + per-head LoRA), SlimSpec 2605.10453, Domino 2605.29707,
 * Basharin et al. 2410.17765. See ~/Projects/speed-plan/research/novelty-review.md.
 */
#include <torch/torch.h>

#include <vector>

namespace olmo_cpp {

class SLMTPHeadImpl : public torch::nn::Module {
 public:
  SLMTPHeadImpl(int64_t d_model, int64_t rank);

  /// u = a h + (h V) U^T, the head's input to the shared LM head.
  torch::Tensor mix(const torch::Tensor& h) const;

  /// U = 0, a = 1, V ~ N(0, std) clipped at 3 std.
  void reset(double std);

  /// T = W diag(g) U [V, r] for drafting, rebuilt only when W, g or U change
  /// (tracked by tensor version counters and storage). No autograd.
  const torch::Tensor& draft_T(const torch::Tensor& W, const torch::Tensor& g);

  torch::Tensor U, V, a;  // [d, r], [d, r], []

 private:
  torch::Tensor T_;
  std::vector<int64_t> T_key_;
};
TORCH_MODULE(SLMTPHead);

struct MTPLosses {
  torch::Tensor main;                // next-token CE
  std::vector<torch::Tensor> heads;  // CE of head k against labels shifted by k+1
};

/// Draft logits for every SL-MTP head at one position without touching W:
///   logits_k = (a_k y + T_k c_k) * rsqrt(mean(u_k^2) + eps),  y = main_logits / rsqrt(mean(h^2) + eps)
/// h: [..., d] backbone output; main_logits: [..., V] = LMHead(h), which the
/// verify pass already computed. Reads K x (V x r) instead of V x d.
std::vector<torch::Tensor> sl_mtp_draft_logits(const torch::Tensor& h, const torch::Tensor& main_logits,
                                               const torch::Tensor& W, const torch::Tensor& g, double eps,
                                               const std::vector<SLMTPHeadImpl*>& heads);

/// Main + SL-MTP losses in factored form (the fast path).
///   h: [B, S, d] backbone output; W: [V, d]; g: [d]; labels: [B, S].
MTPLosses sl_mtp_losses(const torch::Tensor& h, const torch::Tensor& W, const torch::Tensor& g,
                        double eps, const std::vector<const SLMTPHeadImpl*>& heads,
                        const torch::Tensor& labels, int64_t ignore_index);

/// Production loss: main + head_weight * mean_k(head_k), computed in blocks of
/// `chunk_rows` rows with every gradient formed during the forward (the trick
/// fused cross-entropy kernels use), so no [tokens x vocab] tensor outlives a
/// block and memory stays O(chunk x vocab). dW and dh remain ONE merged GEMM per
/// block for the main head and all K heads. Differentiable in h, W, g and each
/// head's U, V, a (backward only scales the stored gradients). Gradients are
/// skipped for inputs that do not require them (a frozen W costs no dW GEMM).
torch::Tensor sl_mtp_loss_chunked(const torch::Tensor& h, const torch::Tensor& W, const torch::Tensor& g,
                                  double eps, const std::vector<SLMTPHeadImpl*>& heads,
                                  const torch::Tensor& labels, int64_t ignore_index, double head_weight,
                                  int64_t chunk_rows = 512);

/// The same losses computed directly: each head runs its own LMHead(u_k), a
/// full d x V GEMM. Reference for tests and the cost benchmark.
MTPLosses sl_mtp_losses_direct(const torch::Tensor& h, const torch::Tensor& W, const torch::Tensor& g,
                               double eps, const std::vector<const SLMTPHeadImpl*>& heads,
                               const torch::Tensor& labels, int64_t ignore_index);

}  // namespace olmo_cpp
