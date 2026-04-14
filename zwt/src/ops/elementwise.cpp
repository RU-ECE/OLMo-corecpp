#include "zwt/ops/elementwise.hpp"
#include "zwt/core/stream.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#ifdef USE_CUDA
#include "zwt/src/ops/kernels.hpp"
#endif

namespace zwt::ops {

namespace {

inline int64_t total(const Tensor& t) { return t.numel(); }

void scale_f32_cpu(float* y, float a, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] *= a;
}

void axpy_f32_cpu(float* y, const float* x, float a, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] += a * x[i];
}

void add_f32_cpu(float* out, const float* a, const float* b, int64_t n) {
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void silu_mul_f32_cpu(float* out, const float* gate, const float* up, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    float g = gate[i];
    float s = g / (1.0f + std::exp(-g));  // silu
    out[i] = s * up[i];
  }
}

void silu_mul_backward_f32_cpu(const float* grad_out, const float* gate,
                               const float* up, float* grad_gate,
                               float* grad_up, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    float g = gate[i];
    float sig = 1.0f / (1.0f + std::exp(-g));
    float silu = g * sig;
    // d silu / dg = sig * (1 + g * (1 - sig))
    float dsilu = sig * (1.0f + g * (1.0f - sig));
    grad_gate[i] = grad_out[i] * up[i] * dsilu;
    grad_up[i]   = grad_out[i] * silu;
  }
}

}  // namespace

void scale(Tensor& y, float alpha) {
  if (y.device().is_cuda()) {
#ifdef USE_CUDA
    if (y.dtype() != DType::BF16) throw std::runtime_error("scale: CUDA bf16 only");
    k::scale_bf16(reinterpret_cast<__nv_bfloat16*>(y.data()), alpha, total(y),
                  reinterpret_cast<cudaStream_t>(compute_stream(y.device()).handle));
    return;
#endif
  }
  if (y.dtype() == DType::F32) {
    scale_f32_cpu(y.as<float>(), alpha, total(y));
    return;
  }
  throw std::runtime_error("scale: unsupported dtype on CPU");
}

void axpy(Tensor& y, const Tensor& x, float alpha) {
  if (y.shape() != x.shape()) throw std::runtime_error("axpy: shape mismatch");
  if (y.device().is_cuda()) {
#ifdef USE_CUDA
    k::axpy_bf16(reinterpret_cast<__nv_bfloat16*>(y.data()),
                 reinterpret_cast<const __nv_bfloat16*>(x.data()),
                 alpha, total(y),
                 reinterpret_cast<cudaStream_t>(compute_stream(y.device()).handle));
    return;
#endif
  }
  if (y.dtype() == DType::F32) {
    axpy_f32_cpu(y.as<float>(), x.as<float>(), alpha, total(y));
    return;
  }
  throw std::runtime_error("axpy: unsupported dtype on CPU");
}

void add(Tensor& out, const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape() || out.shape() != a.shape())
    throw std::runtime_error("add: shape mismatch");
  if (out.device().is_cuda()) {
#ifdef USE_CUDA
    k::add_bf16(reinterpret_cast<__nv_bfloat16*>(out.data()),
                reinterpret_cast<const __nv_bfloat16*>(a.data()),
                reinterpret_cast<const __nv_bfloat16*>(b.data()),
                total(out),
                reinterpret_cast<cudaStream_t>(compute_stream(out.device()).handle));
    return;
#endif
  }
  if (out.dtype() == DType::F32) {
    add_f32_cpu(out.as<float>(), a.as<float>(), b.as<float>(), total(out));
    return;
  }
  throw std::runtime_error("add: unsupported dtype on CPU");
}

void add_bias(Tensor& y, const Tensor& bias) {
  const int64_t cols = bias.numel();
  const int64_t rows = y.numel() / cols;
  if (cols * rows != y.numel())
    throw std::runtime_error("add_bias: bias size does not divide y");
  if (y.device().is_cuda()) {
#ifdef USE_CUDA
    k::add_bias_bf16(reinterpret_cast<__nv_bfloat16*>(y.data()),
                     reinterpret_cast<const __nv_bfloat16*>(bias.data()),
                     rows, cols,
                     reinterpret_cast<cudaStream_t>(compute_stream(y.device()).handle));
    return;
#endif
  }
  if (y.dtype() == DType::F32) {
    float* yp = y.as<float>();
    const float* bp = bias.as<float>();
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) yp[r * cols + c] += bp[c];
    }
    return;
  }
  throw std::runtime_error("add_bias: unsupported dtype on CPU");
}

void transpose_bshd_to_bhsd(const Tensor& in, Tensor& out) {
  if (in.rank() != 4 || out.rank() != 4)
    throw std::runtime_error("transpose_bshd_to_bhsd: rank must be 4");
  const int64_t B = in.dim(0);
  const int64_t S = in.dim(1);
  const int64_t H = in.dim(2);
  const int64_t D = in.dim(3);
  if (out.dim(0) != B || out.dim(1) != H || out.dim(2) != S || out.dim(3) != D)
    throw std::runtime_error("transpose_bshd_to_bhsd: out shape must be [B,H,S,D]");
  if (in.device().is_cuda()) {
#ifdef USE_CUDA
    k::transpose_bshd_bhsd_bf16(
        reinterpret_cast<const __nv_bfloat16*>(in.data()),
        reinterpret_cast<__nv_bfloat16*>(out.data()),
        B, S, H, D,
        reinterpret_cast<cudaStream_t>(compute_stream(in.device()).handle));
    return;
#endif
  }
  if (in.dtype() == DType::F32) {
    const float* x = in.as<float>();
    float* y = out.as<float>();
    for (int64_t b = 0; b < B; ++b)
    for (int64_t s = 0; s < S; ++s)
    for (int64_t h = 0; h < H; ++h)
    for (int64_t d = 0; d < D; ++d)
      y[((b * H + h) * S + s) * D + d] = x[((b * S + s) * H + h) * D + d];
    return;
  }
  throw std::runtime_error("transpose_bshd_to_bhsd: unsupported dtype on CPU");
}

void transpose_bhsd_to_bshd(const Tensor& in, Tensor& out) {
  if (in.rank() != 4 || out.rank() != 4)
    throw std::runtime_error("transpose_bhsd_to_bshd: rank must be 4");
  const int64_t B = in.dim(0);
  const int64_t H = in.dim(1);
  const int64_t S = in.dim(2);
  const int64_t D = in.dim(3);
  if (out.dim(0) != B || out.dim(1) != S || out.dim(2) != H || out.dim(3) != D)
    throw std::runtime_error("transpose_bhsd_to_bshd: out shape must be [B,S,H,D]");
  if (in.device().is_cuda()) {
#ifdef USE_CUDA
    k::transpose_bhsd_bshd_bf16(
        reinterpret_cast<const __nv_bfloat16*>(in.data()),
        reinterpret_cast<__nv_bfloat16*>(out.data()),
        B, S, H, D,
        reinterpret_cast<cudaStream_t>(compute_stream(in.device()).handle));
    return;
#endif
  }
  if (in.dtype() == DType::F32) {
    const float* x = in.as<float>();
    float* y = out.as<float>();
    for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h)
    for (int64_t s = 0; s < S; ++s)
    for (int64_t d = 0; d < D; ++d)
      y[((b * S + s) * H + h) * D + d] = x[((b * H + h) * S + s) * D + d];
    return;
  }
  throw std::runtime_error("transpose_bhsd_to_bshd: unsupported dtype on CPU");
}

void bias_backward(const Tensor& grad_y, Tensor& grad_bias) {
  const int64_t cols = grad_bias.numel();
  const int64_t rows = grad_y.numel() / cols;
  if (cols * rows != grad_y.numel())
    throw std::runtime_error("bias_backward: bias size does not divide grad_y");
  if (grad_y.device().is_cuda()) {
#ifdef USE_CUDA
    if (grad_y.dtype() != DType::BF16 || grad_bias.dtype() != DType::F32)
      throw std::runtime_error("bias_backward: expected bf16 grad_y / fp32 grad_bias");
    k::bias_backward_bf16(reinterpret_cast<const __nv_bfloat16*>(grad_y.data()),
                          grad_bias.as<float>(), rows, cols,
                          reinterpret_cast<cudaStream_t>(
                              compute_stream(grad_y.device()).handle));
    return;
#endif
  }
  if (grad_y.dtype() == DType::F32 && grad_bias.dtype() == DType::F32) {
    const float* gy = grad_y.as<float>();
    float* gb = grad_bias.as<float>();
    for (int64_t c = 0; c < cols; ++c) {
      float acc = 0.f;
      for (int64_t r = 0; r < rows; ++r) acc += gy[r * cols + c];
      gb[c] += acc;
    }
    return;
  }
  throw std::runtime_error("bias_backward: unsupported dtype on CPU");
}

void silu_mul(Tensor& out, const Tensor& gate, const Tensor& up) {
  if (out.shape() != gate.shape() || out.shape() != up.shape())
    throw std::runtime_error("silu_mul: shape mismatch");
  if (out.device().is_cuda()) {
#ifdef USE_CUDA
    k::silu_mul_bf16(reinterpret_cast<__nv_bfloat16*>(out.data()),
                     reinterpret_cast<const __nv_bfloat16*>(gate.data()),
                     reinterpret_cast<const __nv_bfloat16*>(up.data()),
                     total(out),
                     reinterpret_cast<cudaStream_t>(compute_stream(out.device()).handle));
    return;
#endif
  }
  if (out.dtype() == DType::F32) {
    silu_mul_f32_cpu(out.as<float>(), gate.as<float>(), up.as<float>(), total(out));
    return;
  }
  throw std::runtime_error("silu_mul: unsupported dtype on CPU");
}

void silu_mul_backward(const Tensor& grad_out, const Tensor& gate, const Tensor& up,
                       Tensor& grad_gate, Tensor& grad_up) {
  if (grad_out.device().is_cuda()) {
#ifdef USE_CUDA
    k::silu_mul_backward_bf16(
        reinterpret_cast<const __nv_bfloat16*>(grad_out.data()),
        reinterpret_cast<const __nv_bfloat16*>(gate.data()),
        reinterpret_cast<const __nv_bfloat16*>(up.data()),
        reinterpret_cast<__nv_bfloat16*>(grad_gate.data()),
        reinterpret_cast<__nv_bfloat16*>(grad_up.data()),
        total(grad_out),
        reinterpret_cast<cudaStream_t>(compute_stream(grad_out.device()).handle));
    return;
#endif
  }
  if (grad_out.dtype() == DType::F32) {
    silu_mul_backward_f32_cpu(grad_out.as<float>(), gate.as<float>(), up.as<float>(),
                              grad_gate.as<float>(), grad_up.as<float>(),
                              total(grad_out));
    return;
  }
  throw std::runtime_error("silu_mul_backward: unsupported dtype on CPU");
}

// Stub for dropout: deterministic, device-tested in a later iteration.
void dropout(Tensor& y, const Tensor& x, Tensor& mask_u8, float p, uint64_t seed) {
  (void)y; (void)x; (void)mask_u8; (void)p; (void)seed;
  throw std::runtime_error("dropout: not yet implemented (set p=0 in config)");
}

}  // namespace zwt::ops
