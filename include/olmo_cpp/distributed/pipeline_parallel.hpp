#pragma once

/**
 * include/olmo_cpp/distributed/pipeline_parallel.hpp
 *
 * Public API for Pipeline Parallelism (PP). PP partitions the *layer stack*
 * across ranks: each of `world_size` ranks (a "stage") owns a contiguous
 * range of transformer blocks, [start_layer, end_layer). The data flows
 * stage-by-stage in a chain: rank 0 -> 1 -> ... -> W-1 in forward, the
 * reverse direction during backward. Activations and gradients move
 * between adjacent stages via point-to-point send/recv (no collectives;
 * just two-rank messages). With micro-batches, multiple inflight messages
 * keep all stages busy (1F1B or interleaved schedules).
 *
 * Collective ops used:
 *   - p2p send/recv only (rank<->rank+1 for activations and grads).
 *   - No allreduce/allgather/reduce_scatter on the PP axis itself.
 *
 * --- Includes from this project ---
 *   - (none from this project; pure declaration)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/distributed/pipeline_parallel.cpp: implementation (currently
 *     placeholder p2p stubs)
 *   Direct call sites in trainer not located via quick grep.
 *
 * --- Role in training pipeline ---
 *   Used when the model is too deep to fit on one device even after FSDP
 *   sharding, or to compose with TP+DP for very large training runs. The
 *   first stage owns the embedding; the last stage owns the LM head and
 *   the loss.
 */

#include <torch/torch.h>
// c10d::Backend exposes send/recv that we use for activation/gradient passing.
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <vector>
#include <memory>
#include <optional>

namespace olmo_cpp {

/// Pipeline parallelism context.
/// Splits transformer layers into stages; each rank holds a subset of layers.
/// Forward: send activations downstream (rank -> rank+1).
/// Backward: send gradients upstream (rank -> rank-1).
class PipelineParallelContext {
 public:
  /// Create from backend. stage_id in [0, num_stages), num_stages = world_size.
  /// Currently uses an "equal split, one stage per rank" placeholder; real
  /// layer ranges are expected to be set by the caller post-construction.
  static std::optional<PipelineParallelContext> create(
      c10::intrusive_ptr<c10d::Backend> backend);

  /// Layer range for this stage: [start_layer, end_layer).
  int64_t start_layer() const { return start_layer_; }
  int64_t end_layer() const { return end_layer_; }
  int64_t num_layers_this_stage() const { return end_layer_ - start_layer_; }

  /// Send tensor to next stage (rank+1). Used for forward activations.
  void send_to_next_stage(const torch::Tensor& t);
  /// Receive tensor from previous stage (rank-1). Forward activations.
  torch::Tensor recv_from_prev_stage(const torch::TensorOptions& opts);
  /// Send gradient to previous stage (rank-1). Backward path.
  void send_grad_to_prev_stage(const torch::Tensor& grad);
  /// Receive gradient from next stage (rank+1). Backward path.
  torch::Tensor recv_grad_from_next_stage(const torch::TensorOptions& opts);

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  /// First stage owns the embedding/input.
  bool is_first_stage() const { return rank_ == 0; }
  /// Last stage owns the LM head/loss.
  bool is_last_stage() const { return rank_ == world_size_ - 1; }

 private:
  PipelineParallelContext(
      c10::intrusive_ptr<c10d::Backend> backend,
      int rank, int world_size,
      int64_t start_layer, int64_t end_layer);

  c10::intrusive_ptr<c10d::Backend> backend_;  // Provides p2p send/recv.
  int rank_;
  int world_size_;
  int64_t start_layer_;  // Inclusive global layer index.
  int64_t end_layer_;    // Exclusive global layer index.
};

}  // namespace olmo_cpp
