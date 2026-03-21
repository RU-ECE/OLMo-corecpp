#pragma once
#include "olmo_cpp/data/composable/instance_source.hpp"
#include <torch/torch.h>
#include <memory>

namespace olmo_cpp {

/// Composable data loader: wraps an instance source into batched tensors
class ComposableDataLoader {
 public:
  ComposableDataLoader(std::unique_ptr<InstanceSource> source,
                       int64_t batch_size, torch::Device device,
                       int64_t pad_token_id = 0);

  /// Get next batch: {input_ids [B, S], labels [B, S]}
  std::tuple<torch::Tensor, torch::Tensor> next_batch();

  /// Whether more batches are available
  bool has_next() const;

  /// Reset for new epoch
  void reset();

 private:
  std::unique_ptr<InstanceSource> source_;
  int64_t batch_size_;
  torch::Device device_;
  int64_t pad_token_id_;
};

}  // namespace olmo_cpp
