#include "olmo_cpp/data/token_dataset.hpp"
#include "olmo_cpp/seed.hpp"
#include <cnpy.h>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace olmo_cpp {

TokenDataset::TokenDataset(const std::string& path, int64_t seq_len, bool shuffle)
    : seq_len_(seq_len), shuffle_(shuffle), chunk_cursor_(0) {
  cnpy::NpyArray arr = cnpy::npy_load(path);
  size_t num_vals = arr.num_vals;

  // Support uint16, uint32, int32, int64
  if (arr.word_size == 2) {
    const uint16_t* data = arr.data<uint16_t>();
    tokens_.reserve(num_vals);
    for (size_t i = 0; i < num_vals; ++i) {
      tokens_.push_back(static_cast<int64_t>(data[i]));
    }
  } else if (arr.word_size == 4) {
    const uint32_t* data = arr.data<uint32_t>();
    tokens_.reserve(num_vals);
    for (size_t i = 0; i < num_vals; ++i) {
      tokens_.push_back(static_cast<int64_t>(data[i]));
    }
  } else if (arr.word_size == 8) {
    const int64_t* data = arr.data<int64_t>();
    tokens_.reserve(num_vals);
    for (size_t i = 0; i < num_vals; ++i) {
      tokens_.push_back(data[i]);
    }
  } else {
    throw std::runtime_error("TokenDataset: unsupported .npy dtype (word_size=" +
                             std::to_string(arr.word_size) + ")");
  }

  // Each chunk needs seq_len + 1 tokens (seq_len inputs + 1 for the last label)
  num_chunks_ = (static_cast<int64_t>(tokens_.size()) - 1) / seq_len;
  if (num_chunks_ == 0) {
    throw std::runtime_error("TokenDataset: not enough tokens for seq_len=" +
                             std::to_string(seq_len));
  }

  chunk_indices_.resize(static_cast<size_t>(num_chunks_));
  for (int64_t i = 0; i < num_chunks_; ++i) {
    chunk_indices_[i] = i;
  }

  // Pre-create the full token tensor on CPU once
  tokens_tensor_ = torch::from_blob(
      tokens_.data(),
      {static_cast<int64_t>(tokens_.size())},
      torch::TensorOptions().dtype(torch::kInt64)).clone();
}

void TokenDataset::reset_epoch() {
  chunk_cursor_ = 0;
  if (shuffle_) {
    // Use the global seeded RNG for reproducibility (mirrors OLMo-core's
    // seed + dp_rank approach from data_loader.py). Falls back to
    // random_device if seed_all() hasn't been called yet.
    try {
      auto& state = global_seed_state();
      std::shuffle(chunk_indices_.begin(), chunk_indices_.end(), state.rng);
    } catch (const std::runtime_error&) {
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(chunk_indices_.begin(), chunk_indices_.end(), g);
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> TokenDataset::prepare_batch_cpu(int64_t batch_size) {
  // Gather chunk offsets (thread-safe cursor advance)
  std::vector<int64_t> offsets(static_cast<size_t>(batch_size));
  {
    std::lock_guard<std::mutex> lock(cursor_mutex_);
    for (int64_t b = 0; b < batch_size; ++b) {
      if (chunk_cursor_ >= static_cast<size_t>(num_chunks_)) {
        reset_epoch();
      }
      offsets[static_cast<size_t>(b)] = chunk_indices_[chunk_cursor_++] * seq_len_;
    }
  }

  // Build index tensor for gather
  auto idx_opts = torch::TensorOptions().dtype(torch::kInt64);
  auto range = torch::arange(seq_len_, idx_opts);
  auto offset_tensor = torch::from_blob(
      offsets.data(), {batch_size, 1}, idx_opts).clone();

  auto input_indices = offset_tensor + range.unsqueeze(0);
  auto label_indices = offset_tensor + range.unsqueeze(0) + 1;

  auto input = tokens_tensor_.index_select(0, input_indices.reshape(-1)).reshape({batch_size, seq_len_});
  auto labels = tokens_tensor_.index_select(0, label_indices.reshape(-1)).reshape({batch_size, seq_len_});

  return {input, labels};
}

std::tuple<torch::Tensor, torch::Tensor> TokenDataset::get_batch(
    int64_t batch_size,
    torch::Device device) {
  // GPU-resident fast path: pure device-side pointer math, zero H2D
  if (gpu_resident_ && device == resident_device_) {
    return get_batch_gpu(batch_size);
  }

  // If we have a prefetched batch, use it
  if (has_prefetch_ && prefetch_future_.valid()) {
    auto [input, labels] = prefetch_future_.get();
    has_prefetch_ = false;
    // Transfer to target device (may already be on the right device)
    return {input.to(device, /*non_blocking=*/true), labels.to(device, /*non_blocking=*/true)};
  }

  // No prefetch available — prepare synchronously
  auto [input, labels] = prepare_batch_cpu(batch_size);
  return {input.to(device, /*non_blocking=*/true), labels.to(device, /*non_blocking=*/true)};
}

std::tuple<torch::Tensor, torch::Tensor> TokenDataset::get_batch_gpu(int64_t batch_size) {
  // Reshuffle on GPU if we've exhausted all chunks
  if (gpu_cursor_ + batch_size > num_chunks_) {
    if (shuffle_) {
      auto perm = torch::randperm(num_chunks_,
          torch::TensorOptions().dtype(torch::kInt64).device(resident_device_));
      gpu_chunk_indices_ = gpu_chunk_indices_.index_select(0, perm);
    }
    gpu_cursor_ = 0;
  }

  // Grab batch_size chunk offsets (all on GPU)
  auto offsets = gpu_chunk_indices_.narrow(0, gpu_cursor_, batch_size) * seq_len_;
  gpu_cursor_ += batch_size;

  // Build gather indices entirely on GPU
  auto range = torch::arange(seq_len_,
      torch::TensorOptions().dtype(torch::kInt64).device(resident_device_));
  auto input_indices = offsets.unsqueeze(1) + range.unsqueeze(0);   // [B, seq_len]
  auto label_indices = input_indices + 1;

  auto input = gpu_tokens_tensor_.index_select(0, input_indices.reshape(-1))
                                  .reshape({batch_size, seq_len_});
  auto labels = gpu_tokens_tensor_.index_select(0, label_indices.reshape(-1))
                                   .reshape({batch_size, seq_len_});

  return {input, labels};
}

void TokenDataset::to_device(torch::Device device) {
  if (!device.is_cuda()) return;  // Only CUDA gets GPU residency

  try {
    torch::Tensor src_tensor = tokens_tensor_;
    std::vector<int64_t> src_indices = chunk_indices_;

#ifdef USE_CUDA
    // Query free VRAM and cap tokens to what fits.
    // Reserve headroom for model weights, activations, and gradients.
    size_t vram_free = 0, vram_total = 0;
    cudaMemGetInfo(&vram_free, &vram_total);

    // Use at most 25% of free VRAM for data — rest is needed for model/activations/grads
    size_t data_budget = vram_free / 4;
    // Each token = 8 bytes (int64), plus chunk index overhead
    size_t max_gpu_tokens = data_budget / sizeof(int64_t);
    int64_t num_tokens = tokens_tensor_.size(0);

    if (max_gpu_tokens > 0 && static_cast<size_t>(num_tokens) > max_gpu_tokens) {
      int64_t capped = static_cast<int64_t>(max_gpu_tokens);
      std::cerr << "TokenDataset: capping GPU-resident data from " << num_tokens
                << " to " << capped << " tokens (free VRAM: "
                << (vram_free / (1024*1024)) << " MB)\n";
      src_tensor = tokens_tensor_.narrow(0, 0, capped);
      // Recompute chunks for the capped range
      int64_t capped_chunks = (capped - 1) / seq_len_;
      src_indices.resize(static_cast<size_t>(capped_chunks));
      for (int64_t i = 0; i < capped_chunks; ++i) {
        src_indices[static_cast<size_t>(i)] = i;
      }
      num_chunks_ = capped_chunks;
    }
#endif

    // Pin memory for fast initial transfer, then copy to device
    auto pinned = src_tensor.pin_memory();
    gpu_tokens_tensor_ = pinned.to(device);

    // Move chunk indices to GPU
    auto idx_tensor = torch::from_blob(
        src_indices.data(),
        {static_cast<int64_t>(src_indices.size())},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    gpu_chunk_indices_ = idx_tensor.to(device);

    gpu_resident_ = true;
    resident_device_ = device;
    gpu_cursor_ = 0;

    std::cerr << "TokenDataset: GPU-resident with " << gpu_tokens_tensor_.size(0)
              << " tokens on " << device << "\n";
  } catch (const c10::Error&) {
    // Insufficient VRAM — fall back to CPU data path
    std::cerr << "TokenDataset: GPU residency failed, falling back to CPU\n";
    gpu_resident_ = false;
  }
}

void TokenDataset::prefetch_next(int64_t batch_size, torch::Device device) {
  // No-op when GPU-resident (data is already on device)
  if (gpu_resident_ && device == resident_device_) return;

  // Launch async batch preparation on a background thread
  prefetch_device_ = device;
  prefetch_future_ = std::async(std::launch::async, [this, batch_size]() {
    return prepare_batch_cpu(batch_size);
  });
  has_prefetch_ = true;
}

}  // namespace olmo_cpp
