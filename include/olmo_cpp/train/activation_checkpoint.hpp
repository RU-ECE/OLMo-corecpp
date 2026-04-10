#pragma once

#include <torch/torch.h>
#include <functional>

namespace olmo_cpp {

/// Activation checkpointing: trades compute for memory by recomputing
/// forward activations during backward instead of storing them.
class ActivationCheckpoint {
 public:
  /// Checkpoint a function: saves only inputs during forward,
  /// recomputes during backward.
  static torch::Tensor checkpoint(
      std::function<torch::Tensor(torch::Tensor)> fn,
      torch::Tensor input);

  /// Whether to checkpoint a given layer based on interval
  static bool should_checkpoint(int64_t layer_idx, int64_t interval);
};

}  // namespace olmo_cpp
