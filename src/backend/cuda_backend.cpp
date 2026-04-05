#include "olmo_cpp/backend/cuda_backend.hpp"

#ifdef USE_CUDA
#include <torch/library.h>
#endif

namespace olmo_cpp {

// CUDA kernels are compiled for FP32. Under BF16 autocast, inputs arrive as
// BF16. We promote to FP32, run the fused kernel (1 pass), then cast back.
// This is MUCH faster than the fallback (5+ separate ATen kernels in BF16).

torch::Tensor CUDABackend::rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda()) {
    auto orig_dtype = x.scalar_type();
    bool need_cast = (orig_dtype != torch::kFloat32);
    auto x_f32 = need_cast ? x.to(torch::kFloat32) : x;
    auto w_f32 = (weight.defined() && need_cast) ? weight.to(torch::kFloat32) : weight;
    try {
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::rms_norm", "");
      c10::optional<torch::Tensor> w = w_f32.defined()
          ? c10::optional<torch::Tensor>(w_f32) : c10::nullopt;
      auto result = op.typed<torch::Tensor(const torch::Tensor&, const c10::optional<torch::Tensor>&, double)>()
          .call(x_f32, w, eps);
      return need_cast ? result.to(orig_dtype) : result;
    } catch (...) {
      // Fall through to default
    }
  }
#endif
  return IBackend::rms_norm(x, weight, eps);
}

torch::Tensor CUDABackend::silu_mul(torch::Tensor gate, torch::Tensor up) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (gate.is_cuda()) {
    auto orig_dtype = gate.scalar_type();
    bool need_cast = (orig_dtype != torch::kFloat32);
    auto gate_f32 = need_cast ? gate.to(torch::kFloat32) : gate;
    auto up_f32 = need_cast ? up.to(torch::kFloat32) : up;
    try {
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::silu_mul", "");
      auto result = op.typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>()
          .call(gate_f32, up_f32);
      return need_cast ? result.to(orig_dtype) : result;
    } catch (...) {
      // Fall through to default
    }
  }
#endif
  return IBackend::silu_mul(gate, up);
}

torch::Tensor CUDABackend::apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (t.is_cuda()) {
    auto orig_dtype = t.scalar_type();
    bool need_cast = (orig_dtype != torch::kFloat32);
    auto t_f32 = need_cast ? t.to(torch::kFloat32) : t;
    auto sin_f32 = need_cast ? sin.to(torch::kFloat32) : sin;
    auto cos_f32 = need_cast ? cos.to(torch::kFloat32) : cos;
    try {
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::apply_rope", "");
      auto result = op.typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>()
          .call(t_f32, cos_f32, sin_f32);
      return need_cast ? result.to(orig_dtype) : result;
    } catch (...) {
      // Fall through to default
    }
  }
#endif
  return IBackend::apply_rope(t, sin, cos);
}

torch::Tensor CUDABackend::residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                               torch::Tensor weight, double eps) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda()) {
    auto orig_dtype = x.scalar_type();
    bool need_cast = (orig_dtype != torch::kFloat32);
    auto x_f32 = need_cast ? x.to(torch::kFloat32) : x;
    auto res_f32 = need_cast ? residual.to(torch::kFloat32) : residual;
    auto w_f32 = (weight.defined() && need_cast) ? weight.to(torch::kFloat32) : weight;
    try {
      auto op = torch::Dispatcher::singleton()
          .findSchemaOrThrow("olmo_ops::residual_rms_norm", "");
      c10::optional<torch::Tensor> w = w_f32.defined()
          ? c10::optional<torch::Tensor>(w_f32) : c10::nullopt;
      auto results = op.typed<std::vector<torch::Tensor>(
          const torch::Tensor&, const torch::Tensor&,
          const c10::optional<torch::Tensor>&, double)>()
          .call(x_f32, res_f32, w, eps);
      return need_cast ? results[0].to(orig_dtype) : results[0];
    } catch (...) {
      // Fall through to default
    }
  }
#endif
  return IBackend::residual_rms_norm(x, residual, weight, eps);
}

void use_cuda_backend() {
  set_backend(std::make_unique<CUDABackend>());
}

}  // namespace olmo_cpp
