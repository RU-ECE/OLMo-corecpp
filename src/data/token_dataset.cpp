#include "olmo_cpp/data/token_dataset.hpp"
#include <cnpy.h>
#include <algorithm>
#include <random>
#include <stdexcept>

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
  shuffle_ = shuffle;
}

void TokenDataset::reset_epoch() {
  chunk_cursor_ = 0;
  if (shuffle_) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(chunk_indices_.begin(), chunk_indices_.end(), g);
  }
}

std::tuple<torch::Tensor, torch::Tensor> TokenDataset::get_batch(
    int64_t batch_size,
    torch::Device device) {
  auto input_buf = std::vector<int64_t>(static_cast<size_t>(batch_size * seq_len_));
  auto label_buf = std::vector<int64_t>(static_cast<size_t>(batch_size * seq_len_));

  for (int64_t b = 0; b < batch_size; ++b) {
    if (chunk_cursor_ >= static_cast<size_t>(num_chunks_)) {
      reset_epoch();
    }
    int64_t chunk_idx = chunk_indices_[chunk_cursor_++];
    int64_t offset = chunk_idx * seq_len_;

    for (int64_t s = 0; s < seq_len_; ++s) {
      size_t idx = static_cast<size_t>(b * seq_len_ + s);
      input_buf[idx] = tokens_[static_cast<size_t>(offset + s)];
      // Labels = next token (always valid since num_chunks guarantees room)
      label_buf[idx] = tokens_[static_cast<size_t>(offset + s + 1)];
    }
  }

  // Create on CPU first, then move to device (more reliable on MPS)
  auto cpu_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto input = torch::from_blob(input_buf.data(), {batch_size, seq_len_}, cpu_opts).clone().to(device);
  auto labels = torch::from_blob(label_buf.data(), {batch_size, seq_len_}, cpu_opts).clone().to(device);
  return {std::move(input), std::move(labels)};
}

}  // namespace olmo_cpp
