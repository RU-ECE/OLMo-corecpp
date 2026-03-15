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
