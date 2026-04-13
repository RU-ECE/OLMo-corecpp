#include "olmo_cpp/backend/cuda_backend.hpp"

#ifdef USE_CUDA
#include <torch/library.h>
#endif

namespace olmo_cpp {

// Dispatch to custom CUDA kernels which now handle both FP32 and BF16
// natively (BF16 uses FP32 accumulation internally). No dtype promotion
// needed — zero allocation overhead, compatible with CUDA graphs.
//
// Each op resolves its OperatorHandle exactly once on first call (static
// local init is thread-safe since C++11). The old code called
// findSchemaOrThrow() on every invocation, which was a hash-map+string
// lookup per norm/silu/rope per layer per forward — ~100 string lookups
// per step on a 24-layer model. Now those are compile-time-constant
// loads after warmup.

namespace {

inline bool supported_dtype(torch::ScalarType t) {
  return t == torch::kFloat32 || t == torch::kBFloat16;
}

#ifdef OLMO_HAS_CUDA_KERNELS
inline c10::optional<c10::OperatorHandle> resolve_op(const char* name) {
  return c10::Dispatcher::singleton().findOp({name, ""});
}
#endif

}  // namespace

torch::Tensor CUDABackend::rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
#ifdef OLMO_HAS_CUDA_KERNELS
  using FnType =
      torch::Tensor(const torch::Tensor&, const c10::optional<torch::Tensor>&, double);
  static const auto op = resolve_op("olmo_ops::rms_norm");
  if (op.has_value() && x.is_cuda() && supported_dtype(x.scalar_type())) {
    c10::optional<torch::Tensor> w = weight.defined()
        ? c10::optional<torch::Tensor>(weight) : c10::nullopt;
    return op->typed<FnType>().call(x, w, eps);
  }
#endif
  return IBackend::rms_norm(x, weight, eps);
}

torch::Tensor CUDABackend::silu_mul(torch::Tensor gate, torch::Tensor up) {
#ifdef OLMO_HAS_CUDA_KERNELS
  using FnType = torch::Tensor(const torch::Tensor&, const torch::Tensor&);
  static const auto op = resolve_op("olmo_ops::silu_mul");
  if (op.has_value() && gate.is_cuda() && supported_dtype(gate.scalar_type())) {
    return op->typed<FnType>().call(gate, up);
  }
#endif
  return IBackend::silu_mul(gate, up);
}

torch::Tensor CUDABackend::apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos) {
#ifdef OLMO_HAS_CUDA_KERNELS
  using FnType =
      torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
  static const auto op = resolve_op("olmo_ops::apply_rope");
  if (op.has_value() && t.is_cuda() && supported_dtype(t.scalar_type())) {
    // Kernel takes (t, cos, sin) — note the argument order swap.
    return op->typed<FnType>().call(t, cos, sin);
  }
#endif
  return IBackend::apply_rope(t, sin, cos);
}

torch::Tensor CUDABackend::residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                               torch::Tensor weight, double eps) {
#ifdef OLMO_HAS_CUDA_KERNELS
  using FnType = std::vector<torch::Tensor>(
      const torch::Tensor&, const torch::Tensor&,
      const c10::optional<torch::Tensor>&, double);
  static const auto op = resolve_op("olmo_ops::residual_rms_norm");
  if (op.has_value() && x.is_cuda() && supported_dtype(x.scalar_type())) {
    c10::optional<torch::Tensor> w = weight.defined()
        ? c10::optional<torch::Tensor>(weight) : c10::nullopt;
    auto results = op->typed<FnType>().call(x, residual, w, eps);
    return results[0];
  }
#endif
  return IBackend::residual_rms_norm(x, residual, weight, eps);
}

void use_cuda_backend() {
  set_backend(std::make_unique<CUDABackend>());
}

}  // namespace olmo_cpp
