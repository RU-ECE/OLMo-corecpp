/**
 * src/distributed/tensor_parallel.cpp
 *
 * ─── What "Tensor Parallelism" is ───────────────────────────────────
 *
 * TP (Megatron-LM, 2019) splits the *weight matrices themselves*
 * across GPUs. For a Linear layer y = x · W with W of shape [in, out]:
 *
 *   "column-parallel": each rank holds W[:, slice], computes a partial
 *     output [in, out/world_size], and the activations are concatenated
 *     across ranks (all_gather, but typically deferred).
 *
 *   "row-parallel":    each rank holds W[slice, :], computes a partial
 *     sum over a slice of the input dim, and an **all_reduce** sums
 *     the partials to get the true output.
 *
 * In a transformer block the standard recipe is:
 *   QKV projection — column-parallel
 *   attention output projection — row-parallel  (allreduce here)
 *   FFN up/gate — column-parallel
 *   FFN down — row-parallel  (allreduce here)
 *
 * So TP costs two allreduces per block per fwd, two per bwd. That's
 * heavy bandwidth — TP is usually only used inside a single node where
 * NVLink can keep up, and combined with DP/PP across nodes.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/distributed/tensor_parallel.hpp : TP context + helpers.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/attention.cpp / feed_forward.cpp: when a TP context is
 *     present, the linear ops dispatch to TP-aware variants that
 *     allreduce.
 *
 * --- Role in training pipeline ---
 *   Used when a single layer's weights don't fit on one device.
 *   Disabled by default; activated when world_size_tp > 1.
 */
#include "olmo_cpp/distributed/tensor_parallel.hpp"

namespace olmo_cpp {

std::optional<TensorParallelContext> TensorParallelContext::create(
    c10::intrusive_ptr<c10d::Backend> backend) {
  if (!backend) return std::nullopt;
  int world_size = backend->getSize();
  if (world_size < 2) return std::nullopt;
  return TensorParallelContext(backend, backend->getRank(), world_size);
}

TensorParallelContext::TensorParallelContext(
    c10::intrusive_ptr<c10d::Backend> backend, int rank, int world_size)
    : backend_(std::move(backend)), rank_(rank), world_size_(world_size) {}

torch::Tensor TensorParallelContext::column_parallel_linear(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  // x: [B, S, in_features], weight: [out_features_local, in_features]
  // output: [B, S, out_features_local] - sharded along output dim
  return torch::nn::functional::linear(x, weight, bias);
}

torch::Tensor TensorParallelContext::row_parallel_linear(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  // x: [B, S, in_features_local] (sharded), weight: [out_features, in_features_local]
  auto out = torch::nn::functional::linear(x, weight, bias);
  std::vector<at::Tensor> tensors = {out};
  backend_->allreduce(tensors)->wait();
  return out;
}

torch::Tensor TensorParallelContext::allgather_sequence(const torch::Tensor& x) {
  // Allgather: each rank has [B, S_local, D], gather to [B, S_local*W, D]
  // outputTensors[rank][0] = buffer for rank to receive full gathered tensor
  std::vector<int64_t> out_sizes = x.sizes().vec();
  out_sizes[1] *= world_size_;  // sequence dim
  std::vector<std::vector<at::Tensor>> outputs(world_size_);
  for (int r = 0; r < world_size_; ++r) {
    outputs[r] = {torch::empty(out_sizes, x.options())};
  }
  std::vector<at::Tensor> inputs = {x};
  backend_->allgather(outputs, inputs, c10d::AllgatherOptions())->wait();
  return outputs[rank_][0];
}

}  // namespace olmo_cpp
