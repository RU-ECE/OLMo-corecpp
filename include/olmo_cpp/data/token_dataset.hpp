#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include <optional>
#include <future>
#include <mutex>

namespace olmo_cpp {

/// Token dataset backed by .npy file. Loads tokens, chunks into fixed seq_len.
/// Supports uint16 or uint32 token IDs (common for OLMo).
/// Includes async prefetch: while GPU runs forward/backward on batch N,
/// CPU prepares batch N+1 in a background thread.
class TokenDataset {
 public:
  /// Load token array from .npy file. Expects 1D array of token IDs.
  TokenDataset(const std::string& path, int64_t seq_len, bool shuffle = true);

  /// Number of chunks (sequences) in the dataset
  int64_t size() const { return num_chunks_; }

  /// Get a batch of sequences. If prefetch was started, returns the
  /// prefetched batch (zero-wait). Otherwise prepares synchronously.
  std::tuple<torch::Tensor, torch::Tensor> get_batch(
      int64_t batch_size,
      torch::Device device);

  /// Start prefetching the next batch in a background thread.
  /// Call this immediately after get_batch() to overlap data prep with compute.
  void prefetch_next(int64_t batch_size, torch::Device device);

  /// Reset shuffle indices for next epoch
  void reset_epoch();

  /// Move the entire token tensor to the given device (VRAM).
  /// After this call, get_batch() does pure GPU pointer arithmetic
  /// with zero CPU involvement and zero H2D copies per step.
  /// No-op for non-CUDA devices. Falls back to CPU on VRAM allocation failure.
  void to_device(torch::Device device);

  /// Check if the dataset is GPU-resident
  bool is_gpu_resident() const { return gpu_resident_; }

 private:
  /// Prepare a batch on CPU (no device transfer yet)
  std::tuple<torch::Tensor, torch::Tensor> prepare_batch_cpu(int64_t batch_size);

  /// Prepare a batch entirely on GPU (zero CPU involvement)
  std::tuple<torch::Tensor, torch::Tensor> get_batch_gpu(int64_t batch_size);

  std::vector<int64_t> tokens_;
  torch::Tensor tokens_tensor_;  // Pre-built CPU tensor for fast gather
  int64_t seq_len_;
  int64_t num_chunks_;
  bool shuffle_;
  std::vector<int64_t> chunk_indices_;
  size_t chunk_cursor_;

  // Async prefetch state
  std::future<std::tuple<torch::Tensor, torch::Tensor>> prefetch_future_;
  torch::Device prefetch_device_{torch::kCPU};
  bool has_prefetch_ = false;
  std::mutex cursor_mutex_;

  // GPU-resident data state
  torch::Tensor gpu_tokens_tensor_;   // Full token array on CUDA
  torch::Tensor gpu_chunk_indices_;   // Shuffled chunk indices on CUDA
  int64_t gpu_cursor_ = 0;
  bool gpu_resident_ = false;
  torch::Device resident_device_{torch::kCPU};
};

}  // namespace olmo_cpp
