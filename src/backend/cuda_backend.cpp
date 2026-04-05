#include "olmo_cpp/backend/cuda_backend.hpp"

#ifdef USE_CUDA
#include <torch/library.h>
#endif

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// CUDA fused kernels — dispatch to custom .cu kernels for both FP32 and BF16.
// The kernels handle dtype internally (BF16 uses FP32 accumulation).
// ---------------------------------------------------------------------------

torch::Tensor CUDABackend::rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda()) {
    try {
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::rms_norm", "");
      c10::optional<torch::Tensor> w = weight.defined()
          ? c10::optional<torch::Tensor>(weight) : c10::nullopt;
      return op.typed<torch::Tensor(const torch::Tensor&, const c10::optional<torch::Tensor>&, double)>()
          .call(x, w, eps);
    } catch (...) {}
  }
#endif
  return IBackend::rms_norm(x, weight, eps);
}

torch::Tensor CUDABackend::silu_mul(torch::Tensor gate, torch::Tensor up) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (gate.is_cuda()) {
    try {
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::silu_mul", "");
      return op.typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>()
          .call(gate, up);
    } catch (...) {}
  }
#endif
  return IBackend::silu_mul(gate, up);
}

torch::Tensor CUDABackend::apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (t.is_cuda()) {
    try {
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::apply_rope", "");
      return op.typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>()
          .call(t, cos, sin);
    } catch (...) {}
  }
#endif
  return IBackend::apply_rope(t, sin, cos);
}

torch::Tensor CUDABackend::residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                               torch::Tensor weight, double eps) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda()) {
    try {
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::residual_rms_norm", "");
      c10::optional<torch::Tensor> w = weight.defined()
          ? c10::optional<torch::Tensor>(weight) : c10::nullopt;
      auto results = op.typed<std::vector<torch::Tensor>(
          const torch::Tensor&, const torch::Tensor&,
          const c10::optional<torch::Tensor>&, double)>()
          .call(x, residual, w, eps);
      return results[0];
    } catch (...) {}
  }
#endif
  return IBackend::residual_rms_norm(x, residual, weight, eps);
}

void use_cuda_backend() {
  set_backend(std::make_unique<CUDABackend>());
}

}  // namespace olmo_cpp
