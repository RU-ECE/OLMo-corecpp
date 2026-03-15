#pragma once

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <vector>
#include <memory>
#include <optional>

namespace olmo_cpp {

/// Pipeline parallelism context.
/// Splits transformer layers into stages; each rank holds a subset of layers.
/// Forward: send activations downstream; Backward: send gradients upstream.
class PipelineParallelContext {
 public:
  /// Create from backend. stage_id in [0, num_stages), num_stages = world_size.
  static std::optional<PipelineParallelContext> create(
      c10::intrusive_ptr<c10d::Backend> backend);

  /// Layer range for this stage: [start_layer, end_layer)
  int64_t start_layer() const { return start_layer_; }
  int64_t end_layer() const { return end_layer_; }
  int64_t num_layers_this_stage() const { return end_layer_ - start_layer_; }

  /// Send tensor to next stage (rank+1)
  void send_to_next_stage(const torch::Tensor& t);
  /// Receive tensor from previous stage (rank-1)
  torch::Tensor recv_from_prev_stage(const torch::TensorOptions& opts);
  /// Send gradient to previous stage
  void send_grad_to_prev_stage(const torch::Tensor& grad);
  /// Receive gradient from next stage
  torch::Tensor recv_grad_from_next_stage(const torch::TensorOptions& opts);

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  bool is_first_stage() const { return rank_ == 0; }
  bool is_last_stage() const { return rank_ == world_size_ - 1; }

 private:
  PipelineParallelContext(
      c10::intrusive_ptr<c10d::Backend> backend,
      int rank, int world_size,
      int64_t start_layer, int64_t end_layer);

  c10::intrusive_ptr<c10d::Backend> backend_;
  int rank_;
  int world_size_;
  int64_t start_layer_;
  int64_t end_layer_;
};

}  // namespace olmo_cpp
