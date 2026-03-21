#pragma once
#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

enum class PaddingDirection { Left, Right };

/// Data collator: pads and batches sequences
class DataCollator {
 public:
  DataCollator(int64_t pad_token_id = 0,
               PaddingDirection padding_dir = PaddingDirection::Right,
               int64_t max_length = -1);

  /// Collate variable-length sequences into a padded batch
  /// Returns {input_ids [B, max_len], attention_mask [B, max_len], labels [B, max_len]}
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> collate(
      const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>& batch) const;

 private:
  int64_t pad_token_id_;
  PaddingDirection padding_dir_;
  int64_t max_length_;
};

}  // namespace olmo_cpp
