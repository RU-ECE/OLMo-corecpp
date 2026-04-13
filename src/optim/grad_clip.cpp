#include "olmo_cpp/optim/grad_clip.hpp"
#include <ATen/ops/_foreach_norm.h>
#include <ATen/ops/_foreach_mul.h>

namespace olmo_cpp {

torch::Tensor clip_grad_norm_gpu(
    const std::vector<torch::Tensor>& parameters,
    double max_norm,
    double norm_type) {

  // Collect defined gradients into a thread-local scratch vector. Reusing
  // the vector's capacity avoids a per-step heap allocation + N push_backs
  // on the hot training loop (called once per optimizer step).
  static thread_local std::vector<torch::Tensor> grads;
  grads.clear();
  grads.reserve(parameters.size());
  for (const auto& p : parameters) {
    if (p.grad().defined()) {
      grads.push_back(p.grad());
    }
  }

  if (grads.empty()) {
    return torch::zeros({});
  }

  torch::Tensor total_norm;

  if (norm_type == std::numeric_limits<double>::infinity()) {
    // Infinity norm: max of all absolute values
    // _foreach_norm doesn't support inf norm, compute manually on GPU
    std::vector<torch::Tensor> abs_maxes;
    abs_maxes.reserve(grads.size());
    for (auto& g : grads) {
      abs_maxes.push_back(g.abs().max());
    }
    total_norm = torch::stack(abs_maxes).max();
  } else {
    // p-norm (typically 2): use _foreach_norm for batched computation
    auto per_param_norms = at::_foreach_norm(grads, norm_type);

    // Stack and compute aggregate norm — all on GPU
    auto norms_stacked = torch::stack(per_param_norms);
    total_norm = norms_stacked.norm(norm_type);
  }

  // Compute clip coefficient on GPU
  auto clip_coef = max_norm / (total_norm + 1e-6);
  clip_coef = torch::clamp_max(clip_coef, 1.0);

  // Apply clipping to all grads in one fused launch
  at::_foreach_mul_(grads, clip_coef);

  return total_norm;
}

}  // namespace olmo_cpp
