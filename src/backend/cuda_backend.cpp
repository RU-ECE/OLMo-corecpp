#include "olmo_cpp/backend/cuda_backend.hpp"

#ifdef USE_CUDA
#include <torch/library.h>
#endif

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// CUDA fused kernels with automatic BF16/FP16 → FP32 promotion.
// Our .cu kernels operate in FP32; we cast on the boundary so the fused
// kernel still runs (1 fused launch vs N separate ATen ops).
// ---------------------------------------------------------------------------

namespace {

/// Cast tensor to FP32 if it's a reduced-precision dtype, returning
/// the original dtype so the caller can cast back.
inline std::pair<torch::Tensor, at::ScalarType> promote_to_f32(const torch::Tensor& t) {
  auto orig = t.scalar_type();
  if (orig == torch::kFloat32) return {t, orig};
  return {t.to(torch::kFloat32), orig};
}

inline torch::Tensor demote_from_f32(torch::Tensor t, at::ScalarType target) {
  return (target == torch::kFloat32) ? t : t.to(target);
}

}  // namespace

torch::Tensor CUDABackend::rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda()) {
    try {
      auto [x32, orig_dtype] = promote_to_f32(x);
      auto w32 = (weight.defined() && weight.scalar_type() != torch::kFloat32)
          ? weight.to(torch::kFloat32) : weight;
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::rms_norm", "");
      c10::optional<torch::Tensor> w = w32.defined()
          ? c10::optional<torch::Tensor>(w32) : c10::nullopt;
      auto result = op.typed<torch::Tensor(const torch::Tensor&, const c10::optional<torch::Tensor>&, double)>()
          .call(x32, w, eps);
      return demote_from_f32(result, orig_dtype);
    } catch (...) {}
  }
#endif
  return IBackend::rms_norm(x, weight, eps);
}

torch::Tensor CUDABackend::silu_mul(torch::Tensor gate, torch::Tensor up) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (gate.is_cuda()) {
    try {
      auto [g32, orig_dtype] = promote_to_f32(gate);
      auto u32 = (up.scalar_type() == torch::kFloat32) ? up : up.to(torch::kFloat32);
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::silu_mul", "");
      auto result = op.typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>()
          .call(g32, u32);
      return demote_from_f32(result, orig_dtype);
    } catch (...) {}
  }
#endif
  return IBackend::silu_mul(gate, up);
}

torch::Tensor CUDABackend::apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (t.is_cuda()) {
    try {
      auto [t32, orig_dtype] = promote_to_f32(t);
      auto sin32 = (sin.scalar_type() == torch::kFloat32) ? sin : sin.to(torch::kFloat32);
      auto cos32 = (cos.scalar_type() == torch::kFloat32) ? cos : cos.to(torch::kFloat32);
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::apply_rope", "");
      auto result = op.typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>()
          .call(t32, cos32, sin32);
      return demote_from_f32(result, orig_dtype);
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
      auto [x32, orig_dtype] = promote_to_f32(x);
      auto r32 = (residual.scalar_type() == torch::kFloat32) ? residual : residual.to(torch::kFloat32);
      auto w32 = (weight.defined() && weight.scalar_type() != torch::kFloat32)
          ? weight.to(torch::kFloat32) : weight;
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::residual_rms_norm", "");
      c10::optional<torch::Tensor> w = w32.defined()
          ? c10::optional<torch::Tensor>(w32) : c10::nullopt;
      auto results = op.typed<std::vector<torch::Tensor>(
          const torch::Tensor&, const torch::Tensor&,
          const c10::optional<torch::Tensor>&, double)>()
          .call(x32, r32, w, eps);
      return demote_from_f32(results[0], orig_dtype);
    } catch (...) {}
  }
#endif
  return IBackend::residual_rms_norm(x, residual, weight, eps);
}

void use_cuda_backend() {
  set_backend(std::make_unique<CUDABackend>());
}

}  // namespace olmo_cpp
