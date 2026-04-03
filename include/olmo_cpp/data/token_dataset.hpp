#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// Token dataset backed by .npy file. Loads tokens, chunks into fixed seq_len.
/// Supports uint16 or uint32 token IDs (common for OLMo).
class TokenDataset {
 public:
  /// Load token array from .npy file. Expects 1D array of token IDs.
  /// \param path Path to .npy file
  /// \param seq_len Length of each sequence chunk
  /// \param shuffle Whether to shuffle chunk indices each epoch
  TokenDataset(const std::string& path, int64_t seq_len, bool shuffle = true);

  /// Number of chunks (sequences) in the dataset
  int64_t size() const { return num_chunks_; }

  /// Get a batch of sequences. Indices are shuffled if shuffle=true.
  /// \param batch_size Number of sequences per batch
  /// \param device Device to move tensors to
  /// \return Tuple of (input_ids, labels) where labels are input_ids shifted by 1
  std::tuple<torch::Tensor, torch::Tensor> get_batch(
      int64_t batch_size,
      torch::Device device);

  /// Reset shuffle indices for next epoch
  void reset_epoch();

 private:
  std::vector<int64_t> tokens_;
  int64_t seq_len_;
  int64_t num_chunks_;
  bool shuffle_;
  std::vector<int64_t> chunk_indices_;
  size_t chunk_cursor_;

  // Pre-allocated buffers to avoid per-batch heap allocation
  int64_t buf_batch_size_ = 0;
  std::vector<int64_t> input_buf_;
  std::vector<int64_t> label_buf_;
};

}  // namespace olmo_cpp
