#include "olmo_cpp/data/token_dataset.hpp"
#include "olmo_cpp/seed.hpp"
#include <cnpy.h>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace olmo_cpp {

/// Query available system RAM in bytes (cross-platform).
static size_t get_available_ram() {
#if defined(__APPLE__)
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  vm_statistics64_data_t vm;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&vm), &count) == KERN_SUCCESS) {
    return static_cast<size_t>(vm.free_count + vm.inactive_count) * vm_page_size;
  }
#elif defined(__linux__)
  struct sysinfo si;
  if (sysinfo(&si) == 0) {
    return static_cast<size_t>(si.freeram) * si.mem_unit;
  }
#elif defined(_WIN32)
  MEMORYSTATUSEX mem;
  mem.dwLength = sizeof(mem);
  if (GlobalMemoryStatusEx(&mem)) {
    return static_cast<size_t>(mem.ullAvailPhys);
  }
#endif
  return 0;  // unknown — skip capping
}

TokenDataset::TokenDataset(const std::string& path, int64_t seq_len, bool shuffle)
    : seq_len_(seq_len), shuffle_(shuffle), chunk_cursor_(0) {
  cnpy::NpyArray arr = cnpy::npy_load(path);
  size_t num_vals = arr.num_vals;

  // Cap tokens to fit in available RAM (leave 2 GB headroom for model + overhead).
  // Each token costs ~24 bytes: int64 vector + CPU tensor clone + GPU copy.
  constexpr size_t headroom = 2ULL * 1024 * 1024 * 1024;
  constexpr size_t bytes_per_token = 24;
  size_t avail = get_available_ram();
  if (avail > 0) {
    size_t budget = (avail > headroom) ? avail - headroom : avail / 2;
    size_t max_tokens = budget / bytes_per_token;
    if (num_vals > max_tokens) {
      std::cerr << "TokenDataset: capping from " << num_vals
                << " to " << max_tokens
                << " tokens (available RAM: " << (avail / (1024*1024)) << " MB)\n";
      num_vals = max_tokens;
    }
  }

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
    // Pin memory for fast initial transfer, then copy to device
    auto pinned = tokens_tensor_.pin_memory();
    gpu_tokens_tensor_ = pinned.to(device);

    // Move chunk indices to GPU
    auto idx_tensor = torch::from_blob(
        chunk_indices_.data(),
        {static_cast<int64_t>(chunk_indices_.size())},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    gpu_chunk_indices_ = idx_tensor.to(device);

    gpu_resident_ = true;
    resident_device_ = device;
    gpu_cursor_ = 0;
  } catch (const c10::Error&) {
    // Insufficient VRAM — silently fall back to CPU data path
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
