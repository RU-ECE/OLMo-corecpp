#pragma once
#include "olmo_cpp/data/composable/token_source.hpp"
#include "olmo_cpp/data/composable/document_source.hpp"
#include <vector>
#include <memory>
#include <random>

namespace olmo_cpp {

/// An instance is a fixed-length token sequence ready for training
struct Instance {
  std::vector<int64_t> input_ids;
  std::vector<int64_t> labels;  // shifted by 1
};

/// Abstract instance source
class InstanceSource {
 public:
  virtual ~InstanceSource() = default;
  virtual bool has_next() const = 0;
  virtual Instance next() = 0;
  virtual void reset() = 0;
};

/// Concat-and-chunk: concatenate all tokens, chunk into seq_len pieces
class ConcatAndChunkInstanceSource : public InstanceSource {
 public:
  ConcatAndChunkInstanceSource(std::unique_ptr<TokenSource> source, int64_t seq_len);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
 private:
  std::unique_ptr<TokenSource> source_;
  int64_t seq_len_;
  std::vector<int64_t> buffer_;
  size_t buffer_offset_ = 0;  // index-based tracking instead of erase()
};

/// Packing instance source: packs multiple documents into seq_len with padding
class PackingInstanceSource : public InstanceSource {
 public:
  PackingInstanceSource(std::unique_ptr<DocumentSource> source, int64_t seq_len,
                        int64_t pad_token_id = 0, int64_t eos_token_id = -1);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
 private:
  std::unique_ptr<DocumentSource> source_;
  int64_t seq_len_, pad_token_id_, eos_token_id_;
  std::vector<int64_t> buffer_;
  size_t buffer_offset_ = 0;  // index-based tracking instead of erase()
};

/// Random instance source: generates random token sequences (for testing)
class RandomInstanceSource : public InstanceSource {
 public:
  RandomInstanceSource(int64_t seq_len, int64_t vocab_size, int64_t num_instances,
                       int64_t seed = 42);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
 private:
  int64_t seq_len_, vocab_size_, num_instances_;
  int64_t cursor_ = 0;
  std::mt19937 rng_;
  int64_t seed_;
};

/// Mixing instance source: mixes instances from multiple sources by ratio
class MixingInstanceSource : public InstanceSource {
 public:
  MixingInstanceSource(std::vector<std::unique_ptr<InstanceSource>> sources,
                       std::vector<double> ratios, int64_t seed = 42);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
 private:
  std::vector<std::unique_ptr<InstanceSource>> sources_;
  std::vector<double> cumulative_ratios_;
  std::mt19937 rng_;
  int64_t seed_;
};

/// Shuffled instance source: buffers and shuffles instances
class ShuffledInstanceSource : public InstanceSource {
 public:
  ShuffledInstanceSource(std::unique_ptr<InstanceSource> source,
                        int64_t buffer_size = 10000, int64_t seed = 42);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
 private:
  std::unique_ptr<InstanceSource> source_;
  std::vector<Instance> buffer_;
  int64_t buffer_size_;
  size_t cursor_ = 0;
  std::mt19937 rng_;
  int64_t seed_;
  bool filled_ = false;
};

}  // namespace olmo_cpp
