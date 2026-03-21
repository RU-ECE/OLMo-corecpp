#pragma once

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <memory>
#include <optional>

namespace olmo_cpp {

/// Distributed Data Parallel (DDP) helper.
/// When RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT are set, initializes
/// ProcessGroupGloo and provides gradient allreduce. Otherwise no-op.
class DDPContext {
 public:
  /// Initialize from environment (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT).
  /// Returns nullopt if not in distributed mode.
  static std::optional<DDPContext> init_from_env();

  /// Broadcast parameters from rank 0 to all ranks (modifies in place).
  void broadcast_parameters(std::vector<torch::Tensor>& parameters);

  /// Allreduce gradients across all parameters, then divide by world_size.
  void allreduce_gradients(const std::vector<torch::Tensor>& parameters);

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  bool is_distributed() const { return backend_ != nullptr; }

 private:
  DDPContext(c10::intrusive_ptr<c10d::Backend> backend, int rank, int world_size);

  c10::intrusive_ptr<c10d::Backend> backend_;
  int rank_;
  int world_size_;
};

}  // namespace olmo_cpp
