#pragma once
#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

/// GPU-resident gradient clipping. Computes norms and applies clipping
/// entirely on device with no D2H synchronization.
/// Returns total_norm as a device-resident tensor (caller can .item<float>()
/// later for logging, but clipping is already applied).
torch::Tensor clip_grad_norm_gpu(
    const std::vector<torch::Tensor>& parameters,
    double max_norm,
    double norm_type = 2.0);

}  // namespace olmo_cpp
