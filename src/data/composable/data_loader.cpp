#include "olmo_cpp/data/composable/data_loader.hpp"
#include <stdexcept>

namespace olmo_cpp {

ComposableDataLoader::ComposableDataLoader(
    std::unique_ptr<InstanceSource> source, int64_t batch_size,
    torch::Device device, int64_t pad_token_id)
    : source_(std::move(source)),
      batch_size_(batch_size),
      device_(device),
      pad_token_id_(pad_token_id) {
  if (batch_size <= 0) {
    throw std::runtime_error(
        "ComposableDataLoader: batch_size must be positive");
  }
}

bool ComposableDataLoader::has_next() const {
  return source_->has_next();
}

std::tuple<torch::Tensor, torch::Tensor> ComposableDataLoader::next_batch() {
  std::vector<Instance> instances;
  instances.reserve(batch_size_);

  for (int64_t i = 0; i < batch_size_ && source_->has_next(); ++i) {
    instances.push_back(source_->next());
  }

  if (instances.empty()) {
    throw std::runtime_error("ComposableDataLoader: no more batches");
  }

  // Determine sequence length from the first instance
  const int64_t seq_len = static_cast<int64_t>(instances[0].input_ids.size());
  const int64_t actual_batch = static_cast<int64_t>(instances.size());

  // Allocate tensors on CPU first, then move to device
  auto input_ids = torch::full({actual_batch, seq_len}, pad_token_id_,
                               torch::TensorOptions().dtype(torch::kLong));
  auto labels = torch::full({actual_batch, seq_len}, static_cast<int64_t>(-100),
                            torch::TensorOptions().dtype(torch::kLong));

  auto input_accessor = input_ids.accessor<int64_t, 2>();
  auto label_accessor = labels.accessor<int64_t, 2>();

  for (int64_t b = 0; b < actual_batch; ++b) {
    const auto& inst = instances[b];
    const int64_t len = static_cast<int64_t>(inst.input_ids.size());
    for (int64_t s = 0; s < std::min(len, seq_len); ++s) {
      input_accessor[b][s] = inst.input_ids[s];
    }
    const int64_t label_len = static_cast<int64_t>(inst.labels.size());
    for (int64_t s = 0; s < std::min(label_len, seq_len); ++s) {
      label_accessor[b][s] = inst.labels[s];
    }
  }

  return {input_ids.to(device_), labels.to(device_)};
}

void ComposableDataLoader::reset() {
  source_->reset();
}

}  // namespace olmo_cpp
