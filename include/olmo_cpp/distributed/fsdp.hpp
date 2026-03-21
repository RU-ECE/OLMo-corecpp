#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>

#ifdef OLMO_HAS_DDP
#include <torch/csrc/distributed/c10d/Backend.hpp>
#endif

namespace olmo_cpp {

/// Sharding strategy for FSDP
enum class ShardingStrategy {
  FULL_SHARD,    // ZeRO-3: shard params + grads + optimizer states
  SHARD_GRAD_OP, // ZeRO-2: shard grads + optimizer states only
  NO_SHARD       // DDP-like: only allreduce gradients
};

/// FSDP (Fully Sharded Data Parallel) context.
/// Shards model parameters across ranks for memory-efficient distributed training.
class FSDPContext {
 public:
#ifdef OLMO_HAS_DDP
  /// Create FSDP context. Returns nullopt if backend is null or world_size < 2.
  static std::optional<FSDPContext> create(
      c10::intrusive_ptr<c10d::Backend> backend,
      ShardingStrategy strategy = ShardingStrategy::FULL_SHARD);

  /// HSDP variant: separate intra-node (shard) and inter-node (replicate) groups
  static std::optional<FSDPContext> create_hsdp(
      c10::intrusive_ptr<c10d::Backend> intra_backend,
      c10::intrusive_ptr<c10d::Backend> inter_backend,
      ShardingStrategy strategy = ShardingStrategy::FULL_SHARD);
#endif

  /// Initially shard full parameters: each rank keeps 1/N of each param
  void shard_params(std::vector<torch::Tensor>& params);

  /// Unshard (allgather) params for forward pass
  void unshard_params(std::vector<torch::Tensor>& params);

  /// Alias for unshard_params
  void allgather_params(std::vector<torch::Tensor>& params) { unshard_params(params); }

  /// Reduce-scatter gradients after backward
  void reduce_scatter_grads(std::vector<torch::Tensor>& grads);

  /// Re-shard params after forward (free full param memory)
  void reshard_params(std::vector<torch::Tensor>& params);

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  ShardingStrategy strategy() const { return strategy_; }

 private:
#ifdef OLMO_HAS_DDP
  FSDPContext(c10::intrusive_ptr<c10d::Backend> backend,
              c10::intrusive_ptr<c10d::Backend> inter_backend,
              int rank, int world_size, ShardingStrategy strategy);

  c10::intrusive_ptr<c10d::Backend> backend_;       // intra-node for HSDP, or main
  c10::intrusive_ptr<c10d::Backend> inter_backend_;  // inter-node for HSDP (nullable)
#else
  FSDPContext(int rank, int world_size, ShardingStrategy strategy);
#endif
  int rank_;
  int world_size_;
  ShardingStrategy strategy_;

  // Stored local shards (one per original param)
  std::vector<torch::Tensor> sharded_params_;
  // Original shapes for unsharding
  std::vector<std::vector<int64_t>> original_shapes_;
  std::vector<int64_t> original_numels_;
  bool is_sharded_ = false;
};

}  // namespace olmo_cpp
