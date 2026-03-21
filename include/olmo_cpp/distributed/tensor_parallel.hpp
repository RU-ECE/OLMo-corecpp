#pragma once

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <memory>
#include <optional>

namespace olmo_cpp {

/// Tensor parallelism (Megatron-style) context.
/// Splits linear layers across TP ranks: column-parallel (output split),
/// row-parallel (input split).
class TensorParallelContext {
 public:
  /// Create from ProcessGroup backend. Returns nullopt if world_size < 2.
  static std::optional<TensorParallelContext> create(
      c10::intrusive_ptr<c10d::Backend> backend);

  /// Column-parallel linear: output dim split across ranks.
  /// y = x @ W_local (W_local is 1/tp_size of full W)
  torch::Tensor column_parallel_linear(
      const torch::Tensor& x,
      const torch::Tensor& weight,
      const c10::optional<torch::Tensor>& bias = c10::nullopt);

  /// Row-parallel linear: input dim split, allreduce output.
  /// Assumes x is already sharded (from column-parallel). Output is full.
  torch::Tensor row_parallel_linear(
      const torch::Tensor& x,
      const torch::Tensor& weight,
      const c10::optional<torch::Tensor>& bias = c10::nullopt);

  /// Allgather tensors along dim 0 (for sequence parallel).
  torch::Tensor allgather_sequence(const torch::Tensor& x);

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  bool is_tensor_parallel() const { return world_size_ > 1; }

 private:
  TensorParallelContext(c10::intrusive_ptr<c10d::Backend> backend, int rank, int world_size);

  c10::intrusive_ptr<c10d::Backend> backend_;
  int rank_;
  int world_size_;
};

}  // namespace olmo_cpp
