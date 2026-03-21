#pragma once

#include <torch/torch.h>
#include <functional>
#include <memory>

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

/// Custom autograd function implementing gradient checkpointing
class CheckpointFunction : public torch::autograd::Function<CheckpointFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor input,
      std::shared_ptr<std::function<torch::Tensor(torch::Tensor)>> fn);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

}  // namespace olmo_cpp
