#pragma once

/**
 * include/olmo_cpp/distributed/ddp.hpp
 *
 * Public API for Distributed Data Parallel (DDP). DDP replicates the *full*
 * model on every rank and shards the *data*: each rank consumes a different
 * mini-batch slice. After backward, gradients are made bit-identical across
 * ranks via an allreduce-sum followed by division by world_size (i.e.
 * arithmetic mean of per-rank gradients). At init time, parameters are
 * broadcast from rank 0 so all replicas start from identical weights.
 *
 * Collective ops used:
 *   - broadcast(rootRank=0)  -> sync initial parameters
 *   - allreduce(SUM)         -> sum gradients across all ranks
 * Note: divide-by-world_size is done locally after the allreduce; combined
 * this is mathematically equivalent to a MEAN reduction.
 *
 * --- Includes from this project ---
 *   - (none from this project; pure declaration)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/distributed/ddp.cpp: real implementation when Gloo is available
 *   - src/distributed/ddp_stub.cpp: no-op fallback when Gloo is missing
 *   Direct call sites in the trainer not located via quick grep.
 *
 * --- Role in training pipeline ---
 *   The simplest data-parallel strategy. Trainer calls broadcast_parameters
 *   once after model construction, and allreduce_gradients after every
 *   backward, before the optimizer step.
 */

#include <torch/torch.h>
// c10d::Backend is the abstract collective interface; concrete backend is
// ProcessGroupGloo here (CPU/TCP) but could be NCCL on GPU clusters.
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <memory>
#include <optional>

namespace olmo_cpp {

/// Distributed Data Parallel (DDP) helper.
/// When RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT are set, initializes
/// ProcessGroupGloo and provides gradient allreduce. Otherwise no-op.
/// Model is replicated on every rank; data is sharded.
class DDPContext {
 public:
  /// Initialize from environment (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT).
  /// Returns nullopt if any of those vars are missing (single-process mode).
  /// Spins up a TCPStore (rank 0 = server) and a ProcessGroupGloo on top.
  static std::optional<DDPContext> init_from_env();

  /// Broadcast parameters from rank 0 to all ranks (modifies in place).
  /// One broadcast per parameter; rank 0 is rootRank. Ensures every replica
  /// starts the run with bitwise-identical weights.
  void broadcast_parameters(std::vector<torch::Tensor>& parameters);

  /// Allreduce gradients across all parameters, then divide by world_size.
  /// Net effect: each rank's `.grad` becomes the arithmetic mean of all
  /// per-rank gradients (sum-allreduce + local divide).
  void allreduce_gradients(const std::vector<torch::Tensor>& parameters);

  /// This process's rank in [0, world_size).
  int rank() const { return rank_; }
  /// Total number of replicas participating in DDP.
  int world_size() const { return world_size_; }
  /// True iff a real backend was initialised (distributed env was present).
  bool is_distributed() const { return backend_ != nullptr; }

 private:
  // Private ctor; instances are produced by init_from_env().
  DDPContext(c10::intrusive_ptr<c10d::Backend> backend, int rank, int world_size);

  c10::intrusive_ptr<c10d::Backend> backend_;  // c10d collective backend (Gloo).
  int rank_;          // This process's rank.
  int world_size_;    // Total number of ranks.
};

}  // namespace olmo_cpp
