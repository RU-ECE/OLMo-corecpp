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

void TokenDataset::drain_stream_prefetch() {
  if (stream_prefetch_inflight_ && stream_prefetch_future_.valid()) {
    stream_prefetch_future_.wait();
  }
  stream_prefetch_inflight_ = false;
}

void TokenDataset::ensure_stream_buf_capacity(int64_t batch_size) {
  if (stream_cap_b_ >= batch_size) return;
  drain_stream_prefetch();
  auto opts = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
  for (int i = 0; i < 2; ++i) {
    stream_pin_in_[i] = torch::empty({batch_size, seq_len_}, opts);
    stream_pin_la_[i] = torch::empty({batch_size, seq_len_}, opts);
  }
  stream_cap_b_ = batch_size;
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
  if (gpu_resident_) {
    // Re-upload shuffled indices to GPU (full corpus path)
    auto idx_tensor = torch::from_blob(
        chunk_indices_.data(),
        {static_cast<int64_t>(chunk_indices_.size())},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    gpu_chunk_indices_ = idx_tensor.to(resident_device_);
    gpu_cursor_ = 0;
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
  if (gpu_resident_ && device == resident_device_) {
    return get_batch_gpu(batch_size);
  }

  // Pinned double-buffer: overlap CPU gather with previous GPU step
  if (streaming_mode_ && device.is_cuda()) {
    ensure_stream_buf_capacity(batch_size);
    if (stream_prefetch_inflight_) {
      stream_prefetch_future_.wait();
      stream_prefetch_inflight_ = false;
      auto in = stream_pin_in_[stream_last_filled_slot_].to(device, /*non_blocking=*/true);
      auto lab = stream_pin_la_[stream_last_filled_slot_].to(device, /*non_blocking=*/true);
      return {in, lab};
    }
    auto [a, b] = prepare_batch_cpu(batch_size);
    return {a.to(device, /*non_blocking=*/true), b.to(device, /*non_blocking=*/true)};
  }

  if (has_prefetch_ && prefetch_future_.valid()) {
    auto [input, labels] = prefetch_future_.get();
    has_prefetch_ = false;
    return {input.to(device, /*non_blocking=*/true), labels.to(device, /*non_blocking=*/true)};
  }

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

void TokenDataset::to_device(torch::Device device, int64_t max_gpu_tokens) {
  if (!device.is_cuda()) {
    resident_device_ = device;
    return;
  }

  streaming_mode_ = false;
  gpu_resident_ = false;

#ifdef USE_CUDA
  const int64_t num_tokens = tokens_tensor_.size(0);
  const size_t bytes_tokens = static_cast<size_t>(num_tokens) * sizeof(int64_t);
  const size_t bytes_indices = static_cast<size_t>(num_chunks_) * sizeof(int64_t);
  const size_t bytes_needed = bytes_tokens + bytes_indices + (4 << 20);  // +4 MiB slack

  size_t vram_free = 0, vram_total = 0;
  cudaMemGetInfo(&vram_free, &vram_total);
  const size_t budget_default = vram_free / 4;
  size_t budget = budget_default;
  if (max_gpu_tokens > 0) {
    const size_t cap_bytes = static_cast<size_t>(max_gpu_tokens) * sizeof(int64_t) + bytes_indices + (4 << 20);
    budget = std::min(budget_default, cap_bytes);
  }

  const bool force_stream = (max_gpu_tokens == -1);
  const bool over_user_cap = (max_gpu_tokens > 0 && num_tokens > max_gpu_tokens);
  const bool fits_budget = (bytes_needed <= budget * 9 / 10);
  const bool try_full_gpu = !force_stream && !over_user_cap && fits_budget;

  if (!try_full_gpu) {
    try {
      tokens_tensor_ = tokens_tensor_.contiguous().pin_memory();
    } catch (const c10::Error&) {
      // stay unpinned; H2D still works
    }
    streaming_mode_ = true;
    resident_device_ = device;
    std::cerr << "TokenDataset: streaming mode (pinned host + async prefetch), "
              << num_tokens << " tokens; full GPU residency needs ~"
              << (bytes_needed / (1024 * 1024)) << " MiB data + indices\n";
    return;
  }

  try {
    auto pinned = tokens_tensor_.pin_memory();
    gpu_tokens_tensor_ = pinned.to(device);

    auto idx_tensor = torch::from_blob(
        chunk_indices_.data(),
        {static_cast<int64_t>(chunk_indices_.size())},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    gpu_chunk_indices_ = idx_tensor.to(device);

    gpu_resident_ = true;
    resident_device_ = device;
    gpu_cursor_ = 0;

    std::cerr << "TokenDataset: GPU-resident with " << gpu_tokens_tensor_.size(0)
              << " tokens on " << device << "\n";
  } catch (const c10::Error&) {
    std::cerr << "TokenDataset: GPU residency allocation failed; using streaming mode\n";
    try {
      tokens_tensor_ = tokens_tensor_.contiguous().pin_memory();
    } catch (const c10::Error&) {
    }
    streaming_mode_ = true;
    resident_device_ = device;
  }
#else
  (void)max_gpu_tokens;
  resident_device_ = device;
#endif
}

void TokenDataset::prefetch_next(int64_t batch_size, torch::Device device) {
  if (gpu_resident_ && device == resident_device_) return;

  if (streaming_mode_ && device.is_cuda()) {
    ensure_stream_buf_capacity(batch_size);
    stream_last_filled_slot_ = stream_write_slot_;
    stream_write_slot_ = 1 - stream_write_slot_;
    const int slot = stream_last_filled_slot_;
    stream_prefetch_future_ = std::async(std::launch::async, [this, batch_size, slot]() {
      auto [a, b] = prepare_batch_cpu(batch_size);
      stream_pin_in_[slot].copy_(a, /*non_blocking=*/false);
      stream_pin_la_[slot].copy_(b, /*non_blocking=*/false);
    });
    stream_prefetch_inflight_ = true;
    return;
  }

  prefetch_device_ = device;
  prefetch_future_ = std::async(std::launch::async, [this, batch_size]() {
    return prepare_batch_cpu(batch_size);
  });
  has_prefetch_ = true;
}

TokenDataset::~TokenDataset() {
  drain_stream_prefetch();
  if (has_prefetch_ && prefetch_future_.valid()) {
    prefetch_future_.wait();
    has_prefetch_ = false;
  }
}

}  // namespace olmo_cpp
