#pragma once

#include <torch/torch.h>
#include <memory>
#include <utility>

namespace olmo_cpp {

/// Abstract compute backend interface.
/// Methods correspond to fused operation patterns in the transformer.
/// Default implementations fall through to standard ATen ops.
class IBackend {
 public:
  virtual ~IBackend() = default;
  virtual const char* name() const = 0;

  // ---- Fused operations ----

  /// Fused RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
  /// Default: 4 separate ATen kernels. Fused: single pass.
  virtual torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps);

  /// Fused SwiGLU elementwise: silu(gate) * up
  /// Called between the GEMMs in FFN. Default: silu kernel + mul kernel.
  virtual torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up);

  /// Fused RoPE apply: t * cos + rotate_half(t) * sin, zero temporaries.
  /// Default: chunk + cat + 2 muls + add (4 temporaries).
  virtual torch::Tensor apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos);

  /// Fused residual + RMSNorm: returns norm(x + residual) * weight
  /// Saves one full d_model read/write vs separate add then norm.
  virtual torch::Tensor residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                           torch::Tensor weight, double eps);

  /// Fused RMSNorm + residual add: returns residual + norm(x) * weight
  /// One pass over x and residual instead of two (norm then add).
  virtual torch::Tensor rms_norm_add(torch::Tensor x, torch::Tensor residual,
                                      torch::Tensor weight, double eps);

  /// Arena memory hints (no-ops by default)
  virtual void begin_scope() {}
  virtual void end_scope() {}
};

/// Global backend accessor. Returns the currently active backend.
IBackend& get_backend();

/// Set a custom backend. Ownership transfers. Passing nullptr resets to default.
void set_backend(std::unique_ptr<IBackend> backend);

}  // namespace olmo_cpp
