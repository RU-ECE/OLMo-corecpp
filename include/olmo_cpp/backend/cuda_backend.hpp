#pragma once

#include "olmo_cpp/backend/backend.hpp"

namespace olmo_cpp {

/// CUDA backend that dispatches to fused CUDA kernels.
/// Auto-activated when a CUDA device is selected.
/// Falls back to default ATen ops for any operation without a fused kernel.
class CUDABackend : public IBackend {
 public:
  const char* name() const override { return "cuda_fused"; }

  /// Fused RMSNorm via custom CUDA kernel (vectorized, warp reductions)
  torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps) override;

  /// Fused SiLU(gate) * up in single kernel (eliminates intermediate alloc)
  torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up) override;

  /// Fused RoPE via custom CUDA kernel
  torch::Tensor apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos) override;

  /// Fused residual add + RMSNorm in single kernel (saves full d_model read/write)
  torch::Tensor residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                   torch::Tensor weight, double eps) override;
};

/// Activate the CUDA fused backend globally
void use_cuda_backend();

}  // namespace olmo_cpp
