/**
 * src/optim/foreach_adamw.cpp
 *
 * Batched AdamW. Implements the standard Loshchilov-Hutter update via
 * _foreach_* fused tensor-list ops so the optimizer step is O(1) kernel
 * launches in the number of parameters instead of O(N).
 *
 * Update (per parameter, per step):
 *     p     ← (1 - lr*wd) * p              (decoupled weight decay first)
 *     m     ← beta1 * m + (1 - beta1) * g
 *     v     ← beta2 * v + (1 - beta2) * g^2
 *     denom ← sqrt(v) + eps * sqrt(1-beta2^t)
 *     p    -= (lr * sqrt(1-beta2^t) / (1 - beta1^t)) * m / denom
 * The bias-correction factor sqrt(1-beta2^t) is folded into both step_size
 * and eps to remove one launch.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/optim/foreach_adamw.hpp: declares ForeachAdamW, options, and
 *     scratch-vector members reused here.
 *
 * --- ATen foreach ops ---
 *   - _foreach_add / _foreach_mul / _foreach_addcmul / _foreach_addcdiv /
 *     _foreach_sqrt: tensor-list versions of the elementwise ops; each
 *     issues a single fused CUDA kernel that processes all tensors in the
 *     list concurrently.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: instantiated when cfg.optimizer == "adamw" (default).
 *
 * --- Role in training pipeline ---
 *   The default optimizer for OLMo C++ training. Constructed once after
 *   model init by train_loop; .step() is called each microbatch after
 *   backward + optional grad clip.
 */
#include "olmo_cpp/optim/foreach_adamw.hpp"
#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_addcmul.h>
#include <ATen/ops/_foreach_addcdiv.h>
#include <ATen/ops/_foreach_sqrt.h>
#include <ATen/ops/_fused_adamw.h>
#include <ATen/Functions.h>
#include <cmath>

namespace olmo_cpp {

namespace {

/// Per-parameter state for ForeachAdamW. Only the two moments live here;
/// the step counter is global (step_count_) because the bias correction
/// is identical for every parameter under AdamW.
struct ForeachAdamWParamState
    : public torch::optim::OptimizerCloneableParamState<ForeachAdamWParamState> {
  TORCH_ARG(torch::Tensor, exp_avg);     ///< First moment m, same shape as param.
  TORCH_ARG(torch::Tensor, exp_avg_sq);  ///< Second moment v, same shape as param.

  /// Save state to a checkpoint archive.
  void serialize(torch::serialize::OutputArchive& archive) const override {
    if (exp_avg().defined()) archive.write("exp_avg", exp_avg());
    if (exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq());
  }

  /// Load state from a checkpoint archive (tolerant of missing keys).
  void serialize(torch::serialize::InputArchive& archive) override {
    torch::Tensor t;
    if (archive.try_read("exp_avg", t)) exp_avg(t);
    if (archive.try_read("exp_avg_sq", t)) exp_avg_sq(t);
  }
};

}  // namespace

ForeachAdamW::ForeachAdamW(std::vector<torch::Tensor> params, ForeachAdamWOptions defaults)
    : Optimizer(
          {torch::optim::OptimizerParamGroup(std::move(params))},
          std::make_unique<ForeachAdamWOptions>(defaults)) {}

ForeachAdamW::ForeachAdamW(std::vector<torch::optim::OptimizerParamGroup> param_groups,
                           ForeachAdamWOptions defaults)
    : Optimizer(
          std::move(param_groups),
          std::make_unique<ForeachAdamWOptions>(defaults)) {}

/// One batched AdamW update across all param groups.
torch::Tensor ForeachAdamW::step(LossClosure closure) {
  // Optimizer body runs without autograd tracking — none of these tensor
  // mutations should appear in the autograd graph.
  torch::NoGradGuard no_grad;
  torch::Tensor loss = {};
  if (closure) {
    // The optional closure may want gradients (re-running fwd+loss).
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  // Pre-increment so step_count_ holds the t value we are about to apply.
  step_count_++;

  // Collect all params, grads, and state tensors into parallel vectors
  // across all param groups (typically just one group for AdamW)
  for (auto& group : param_groups_) {
    auto& options = static_cast<ForeachAdamWOptions&>(group.options());
    const double lr = options.lr();
    const double beta1 = options.beta1();
    const double beta2 = options.beta2();
    const double eps = options.eps();
    const double weight_decay = options.weight_decay();

    // Reuse the scratch vectors across steps — clear() retains capacity so
    // we pay for one allocation per vector on the first step, zero on every
    // step thereafter.
    auto& params_vec = params_scratch_;
    auto& grads_vec = grads_scratch_;
    auto& exp_avg_vec = exp_avg_scratch_;
    auto& exp_avg_sq_vec = exp_avg_sq_scratch_;
    params_vec.clear();
    grads_vec.clear();
    exp_avg_vec.clear();
    exp_avg_sq_vec.clear();
    const size_t n_params = group.params().size();
    params_vec.reserve(n_params);
    grads_vec.reserve(n_params);
    exp_avg_vec.reserve(n_params);
    exp_avg_sq_vec.reserve(n_params);

    for (auto& p : group.params()) {
      if (!p.grad().defined()) continue;

      auto key = p.unsafeGetTensorImpl();

      // One hash lookup per param per step (steady state). The old code did
      // find()+insert() on first step and operator[] on every subsequent step,
      // burning two lookups per param every step.
      auto it = state_.find(key);
      if (it == state_.end()) {
        auto s = std::make_unique<ForeachAdamWParamState>();
        s->exp_avg(torch::zeros_like(p.data()));
        s->exp_avg_sq(torch::zeros_like(p.data()));
        it = state_.emplace(key, std::move(s)).first;
      }

      auto& state = static_cast<ForeachAdamWParamState&>(*it->second);
      params_vec.push_back(p.data());
      grads_vec.push_back(p.grad());
      exp_avg_vec.push_back(state.exp_avg());
      exp_avg_sq_vec.push_back(state.exp_avg_sq());
    }

    if (params_vec.empty()) continue;

    // Item P: when available, dispatch the whole AdamW step (param decay,
    // moment updates, denom, parameter update) as a single fused CUDA
    // kernel via at::_fused_adamw_. Replaces the seven _foreach_*
    // launches below with one. PyTorch ships this for CUDA-resident
    // tensors only; falls back to the unfused path otherwise.
#if !defined(OLMO_DISABLE_FUSED_ADAMW)
    if (!params_vec.empty() && params_vec.front().is_cuda()) {
      // Per-param step counter tensor. _fused_adamw_ wants TensorList of
      // 0-D step tensors. We share a single step value across all params
      // by reusing one tensor (the API still accepts a list).
      // torch::full constructs the 0-D step scalar directly on the device
      // (fill kernel) — avoids the per-step 4-byte H->D that torch::tensor(x)
      // followed by .to(device) would incur.
      auto step_t = torch::full({}, static_cast<double>(step_count_),
                                 torch::TensorOptions().dtype(torch::kFloat32)
                                     .device(params_vec.front().device()));
      std::vector<at::Tensor> step_list(params_vec.size(), step_t);
      std::vector<at::Tensor> max_exp_avg_sq_empty;  // amsgrad=false; unused
      at::_fused_adamw_(
          params_vec, grads_vec, exp_avg_vec, exp_avg_sq_vec,
          max_exp_avg_sq_empty,
          step_list,
          /*lr=*/lr,
          /*beta1=*/beta1,
          /*beta2=*/beta2,
          /*weight_decay=*/weight_decay,
          /*eps=*/eps,
          /*amsgrad=*/false,
          /*maximize=*/false,
          /*grad_scale=*/c10::nullopt,
          /*found_inf=*/c10::nullopt);
      continue;
    }
#endif

    // Bias correction factors. Maintain beta1^t / beta2^t incrementally
    // instead of calling std::pow(beta, step_count_) on every step. Two
    // wrinkles: (a) on the very first step (step_count_ == 1) the running
    // power must be beta^1, so we multiply *before* using; (b) if the user
    // mutates the optimizer's beta values mid-run we have to rebuild from
    // scratch because the running product is otherwise stale.
    if (beta1 != last_beta1_ || beta2 != last_beta2_) {
      beta1_pow_ = std::pow(beta1, static_cast<double>(step_count_));
      beta2_pow_ = std::pow(beta2, static_cast<double>(step_count_));
      last_beta1_ = beta1;
      last_beta2_ = beta2;
    } else {
      beta1_pow_ *= beta1;
      beta2_pow_ *= beta2;
    }
    const double bc1      = 1.0 - beta1_pow_;
    const double bc2      = 1.0 - beta2_pow_;
    const double bc2_sqrt = std::sqrt(bc2);
    // Fold bc2_sqrt into step_size and eps to eliminate one kernel launch:
    //   p -= (lr/bc1) * m / (sqrt(v)/bc2_sqrt + eps)
    // = p -= (lr*bc2_sqrt/bc1) * m / (sqrt(v) + eps*bc2_sqrt)
    double step_size = -lr * bc2_sqrt / bc1;
    double eps_scaled = eps * bc2_sqrt;

    // === Fused operations: 7 kernel launches (6 without weight decay) ===

    // 1. Decoupled weight decay: p *= (1 - lr * weight_decay)
    if (weight_decay != 0.0) {
      at::_foreach_mul_(params_vec, 1.0 - lr * weight_decay);
    }

    // 2-3. Update biased first moment: m = beta1 * m + (1 - beta1) * g
    at::_foreach_mul_(exp_avg_vec, beta1);
    at::_foreach_add_(exp_avg_vec, grads_vec, 1.0 - beta1);

    // 4-5. Update biased second moment: v = beta2 * v + (1 - beta2) * g^2
    at::_foreach_mul_(exp_avg_sq_vec, beta2);
    at::_foreach_addcmul_(exp_avg_sq_vec, grads_vec, grads_vec, 1.0 - beta2);

    // 6. Compute denominator: denom = sqrt(v) + eps * bc2_sqrt
    //    (bc2_sqrt is folded into step_size, eliminating the division kernel)
    auto denom = at::_foreach_sqrt(exp_avg_sq_vec);
    at::_foreach_add_(denom, eps_scaled);

    // 7. Apply update: p += step_size * m / denom
    at::_foreach_addcdiv_(params_vec, exp_avg_vec, denom, step_size);
  }

  return loss;
}

}  // namespace olmo_cpp
