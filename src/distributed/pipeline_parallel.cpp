/**
 * src/distributed/pipeline_parallel.cpp
 *
 * ─── What "Pipeline Parallelism" is ─────────────────────────────────
 *
 * Suppose you have 64 transformer layers and 4 GPUs. Pipeline
 * Parallelism (PP) assigns layers 0-15 to GPU 0, 16-31 to GPU 1, etc.
 * A microbatch goes through GPU 0 (layers 0-15), then its output
 * activations are sent to GPU 1 (layers 16-31), and so on, like an
 * assembly line.
 *
 * The naive version wastes 75% of GPU time (only one stage works at
 * any moment). The classic fix is to split the global batch into
 * many small microbatches and pipeline them — while GPU 1 is working
 * on microbatch i, GPU 0 already started microbatch i+1. The
 * standard schedule is "1F1B" (one-forward-one-backward).
 *
 * Communication is point-to-point send/recv (NOT a collective).
 * That's why PP often pairs well with TP/DP: it doesn't tax the
 * cross-GPU all-reduce bandwidth.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/distributed/pipeline_parallel.hpp : the PP context type.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: when a PP context is constructed, the train loop
 *     interleaves microbatch fwd/bwd via PipelineParallelContext.
 *
 * --- Role in training pipeline ---
 *   Used only for very large models where activation memory pressure
 *   forces splitting the layer stack across devices. Off by default.
 */
#include "olmo_cpp/distributed/pipeline_parallel.hpp"

namespace olmo_cpp {

std::optional<PipelineParallelContext> PipelineParallelContext::create(
    c10::intrusive_ptr<c10d::Backend> backend) {
  if (!backend) return std::nullopt;
  int world_size = backend->getSize();
  if (world_size < 2) return std::nullopt;
  int rank = backend->getRank();
  // Placeholder: assume equal layer split. Actual layer count comes from config.
  // For now we just store rank/world_size; layer ranges set by caller.
  return PipelineParallelContext(backend, rank, world_size, rank, rank + 1);
}

PipelineParallelContext::PipelineParallelContext(
    c10::intrusive_ptr<c10d::Backend> backend,
    int rank, int world_size,
    int64_t start_layer, int64_t end_layer)
    : backend_(std::move(backend)),
      rank_(rank),
      world_size_(world_size),
      start_layer_(start_layer),
      end_layer_(end_layer) {}

void PipelineParallelContext::send_to_next_stage(const torch::Tensor& /*t*/) {
  // TODO: point-to-point send to rank+1
}

torch::Tensor PipelineParallelContext::recv_from_prev_stage(const torch::TensorOptions& opts) {
  // TODO: recv from rank-1
  (void)opts;
  return torch::Tensor();
}

void PipelineParallelContext::send_grad_to_prev_stage(const torch::Tensor& /*grad*/) {
  // TODO: send gradient to rank-1
}

torch::Tensor PipelineParallelContext::recv_grad_from_next_stage(const torch::TensorOptions& opts) {
  // TODO: recv gradient from rank+1
  (void)opts;
  return torch::Tensor();
}

}  // namespace olmo_cpp
