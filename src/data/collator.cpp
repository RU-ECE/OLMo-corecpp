#include "olmo_cpp/data/collator.hpp"
#include <algorithm>
#include <stdexcept>

namespace olmo_cpp {

DataCollator::DataCollator(int64_t pad_token_id, PaddingDirection padding_dir,
                           int64_t max_length)
    : pad_token_id_(pad_token_id),
      padding_dir_(padding_dir),
      max_length_(max_length) {}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> DataCollator::collate(
    const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>&
        batch) const {
  if (batch.empty()) {
    throw std::runtime_error("DataCollator::collate: empty batch");
  }

  // Find the maximum length in the batch
  int64_t max_len = 0;
  for (const auto& [input_ids, labels] : batch) {
    max_len = std::max(max_len, static_cast<int64_t>(input_ids.size()));
  }

  // Apply max_length cap if set
  if (max_length_ > 0) {
    max_len = std::min(max_len, max_length_);
  }

  const int64_t batch_size = static_cast<int64_t>(batch.size());

  // Initialize tensors
  auto input_ids_tensor = torch::full(
      {batch_size, max_len}, pad_token_id_,
      torch::TensorOptions().dtype(torch::kLong));
  auto attention_mask_tensor = torch::zeros(
      {batch_size, max_len},
      torch::TensorOptions().dtype(torch::kLong));
  auto labels_tensor = torch::full(
      {batch_size, max_len}, static_cast<int64_t>(-100),
      torch::TensorOptions().dtype(torch::kLong));

  auto input_acc = input_ids_tensor.accessor<int64_t, 2>();
  auto mask_acc = attention_mask_tensor.accessor<int64_t, 2>();
  auto label_acc = labels_tensor.accessor<int64_t, 2>();

  for (int64_t b = 0; b < batch_size; ++b) {
    const auto& [input_ids, labels] = batch[b];
    const int64_t seq_len = std::min(static_cast<int64_t>(input_ids.size()),
                                     max_len);
    const int64_t label_len = std::min(static_cast<int64_t>(labels.size()),
                                       max_len);

    if (padding_dir_ == PaddingDirection::Right) {
      // Real tokens at the beginning, padding at the end
      for (int64_t s = 0; s < seq_len; ++s) {
        input_acc[b][s] = input_ids[s];
        mask_acc[b][s] = 1;
      }
      for (int64_t s = 0; s < label_len; ++s) {
        label_acc[b][s] = labels[s];
      }
    } else {
      // PaddingDirection::Left: padding at the beginning, real tokens at the end
      int64_t offset = max_len - seq_len;
      for (int64_t s = 0; s < seq_len; ++s) {
        input_acc[b][offset + s] = input_ids[s];
        mask_acc[b][offset + s] = 1;
      }
      int64_t label_offset = max_len - label_len;
      for (int64_t s = 0; s < label_len; ++s) {
        label_acc[b][label_offset + s] = labels[s];
      }
    }
  }

  return {input_ids_tensor, attention_mask_tensor, labels_tensor};
}

}  // namespace olmo_cpp
