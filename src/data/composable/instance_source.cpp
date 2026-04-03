#include "olmo_cpp/data/composable/instance_source.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// ConcatAndChunkInstanceSource
// ---------------------------------------------------------------------------

ConcatAndChunkInstanceSource::ConcatAndChunkInstanceSource(
    std::unique_ptr<TokenSource> source, int64_t seq_len)
    : source_(std::move(source)), seq_len_(seq_len) {
  if (seq_len <= 0) {
    throw std::runtime_error(
        "ConcatAndChunkInstanceSource: seq_len must be positive");
  }
}

bool ConcatAndChunkInstanceSource::has_next() const {
  // We need seq_len + 1 tokens to produce one instance (input + 1 shifted label)
  int64_t available = static_cast<int64_t>(buffer_.size() - buffer_offset_);
  return (available >= seq_len_ + 1) || source_->has_next();
}

Instance ConcatAndChunkInstanceSource::next() {
  // Fill buffer until we have at least seq_len + 1 tokens past offset
  while (static_cast<int64_t>(buffer_.size() - buffer_offset_) < seq_len_ + 1 &&
         source_->has_next()) {
    buffer_.push_back(source_->next());
  }

  int64_t available = static_cast<int64_t>(buffer_.size() - buffer_offset_);
  if (available < seq_len_ + 1) {
    throw std::runtime_error(
        "ConcatAndChunkInstanceSource: not enough tokens for an instance");
  }

  Instance inst;
  auto start = buffer_.begin() + static_cast<ptrdiff_t>(buffer_offset_);
  inst.input_ids.assign(start, start + seq_len_);
  inst.labels.assign(start + 1, start + seq_len_ + 1);

  // Advance offset instead of erasing
  buffer_offset_ += static_cast<size_t>(seq_len_);

  // Compact when offset exceeds half the buffer to bound memory growth
  if (buffer_offset_ > buffer_.size() / 2) {
    buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<ptrdiff_t>(buffer_offset_));
    buffer_offset_ = 0;
  }

  return inst;
}

void ConcatAndChunkInstanceSource::reset() {
  source_->reset();
  buffer_.clear();
  buffer_offset_ = 0;
}

// ---------------------------------------------------------------------------
// PackingInstanceSource
// ---------------------------------------------------------------------------

PackingInstanceSource::PackingInstanceSource(
    std::unique_ptr<DocumentSource> source, int64_t seq_len,
    int64_t pad_token_id, int64_t eos_token_id)
    : source_(std::move(source)),
      seq_len_(seq_len),
      pad_token_id_(pad_token_id),
      eos_token_id_(eos_token_id) {
  if (seq_len <= 0) {
    throw std::runtime_error(
        "PackingInstanceSource: seq_len must be positive");
  }
}

bool PackingInstanceSource::has_next() const {
  return (buffer_.size() - buffer_offset_) > 0 || source_->has_next();
}

Instance PackingInstanceSource::next() {
  // Pack documents into buffer until we have at least seq_len + 1 tokens past offset
  while (static_cast<int64_t>(buffer_.size() - buffer_offset_) < seq_len_ + 1 &&
         source_->has_next()) {
    Document doc = source_->next();
    if (doc.tokens.empty()) continue;

    // If buffer has content and we have an eos_token_id, add separator
    if (buffer_.size() > buffer_offset_ && eos_token_id_ >= 0) {
      buffer_.push_back(eos_token_id_);
    }

    buffer_.insert(buffer_.end(), doc.tokens.begin(), doc.tokens.end());
  }

  int64_t available = static_cast<int64_t>(buffer_.size() - buffer_offset_);
  if (available == 0) {
    throw std::runtime_error("PackingInstanceSource: no more instances");
  }

  Instance inst;
  auto start = buffer_.begin() + static_cast<ptrdiff_t>(buffer_offset_);

  if (available >= seq_len_ + 1) {
    inst.input_ids.assign(start, start + seq_len_);
    inst.labels.assign(start + 1, start + seq_len_ + 1);
    buffer_offset_ += static_cast<size_t>(seq_len_);
  } else {
    int64_t real_len = available - 1;
    if (real_len <= 0) {
      inst.input_ids.resize(seq_len_, pad_token_id_);
      inst.labels.resize(seq_len_, -100);
      buffer_offset_ = buffer_.size();
    } else {
      inst.input_ids.assign(start, start + real_len);
      inst.labels.assign(start + 1, start + real_len + 1);

      while (static_cast<int64_t>(inst.input_ids.size()) < seq_len_) {
        inst.input_ids.push_back(pad_token_id_);
        inst.labels.push_back(-100);
      }
      buffer_offset_ = buffer_.size();
    }
  }

  // Compact when offset exceeds half the buffer to bound memory growth
  if (buffer_offset_ > buffer_.size() / 2) {
    buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<ptrdiff_t>(buffer_offset_));
    buffer_offset_ = 0;
  }

  return inst;
}

void PackingInstanceSource::reset() {
  source_->reset();
  buffer_.clear();
  buffer_offset_ = 0;
}

// ---------------------------------------------------------------------------
// RandomInstanceSource
// ---------------------------------------------------------------------------

RandomInstanceSource::RandomInstanceSource(int64_t seq_len, int64_t vocab_size,
                                           int64_t num_instances, int64_t seed)
    : seq_len_(seq_len),
      vocab_size_(vocab_size),
      num_instances_(num_instances),
      rng_(static_cast<unsigned>(seed)),
      seed_(seed) {}

bool RandomInstanceSource::has_next() const {
  return cursor_ < num_instances_;
}

Instance RandomInstanceSource::next() {
  if (!has_next()) {
    throw std::runtime_error("RandomInstanceSource: no more instances");
  }

  std::uniform_int_distribution<int64_t> dist(0, vocab_size_ - 1);

  Instance inst;
  // Generate seq_len + 1 tokens, then split into input/labels
  std::vector<int64_t> tokens(seq_len_ + 1);
  for (auto& t : tokens) {
    t = dist(rng_);
  }
  inst.input_ids.assign(tokens.begin(), tokens.begin() + seq_len_);
  inst.labels.assign(tokens.begin() + 1, tokens.begin() + seq_len_ + 1);

  ++cursor_;
  return inst;
}

void RandomInstanceSource::reset() {
  cursor_ = 0;
  rng_.seed(static_cast<unsigned>(seed_));
}

// ---------------------------------------------------------------------------
// MixingInstanceSource
// ---------------------------------------------------------------------------

MixingInstanceSource::MixingInstanceSource(
    std::vector<std::unique_ptr<InstanceSource>> sources,
    std::vector<double> ratios, int64_t seed)
    : sources_(std::move(sources)),
      rng_(static_cast<unsigned>(seed)),
      seed_(seed) {
  if (sources_.empty()) {
    throw std::runtime_error("MixingInstanceSource: no sources provided");
  }
  if (sources_.size() != ratios.size()) {
    throw std::runtime_error(
        "MixingInstanceSource: sources and ratios size mismatch");
  }

  double sum = std::accumulate(ratios.begin(), ratios.end(), 0.0);
  if (sum <= 0.0) {
    throw std::runtime_error(
        "MixingInstanceSource: ratios must sum to positive value");
  }

  cumulative_ratios_.resize(ratios.size());
  double cumulative = 0.0;
  for (size_t i = 0; i < ratios.size(); ++i) {
    cumulative += ratios[i] / sum;
    cumulative_ratios_[i] = cumulative;
  }
  cumulative_ratios_.back() = 1.0;
}

bool MixingInstanceSource::has_next() const {
  for (const auto& s : sources_) {
    if (s->has_next()) {
      return true;
    }
  }
  return false;
}

Instance MixingInstanceSource::next() {
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (size_t attempt = 0; attempt < sources_.size() * 10; ++attempt) {
    double r = dist(rng_);
    size_t idx = 0;
    while (idx < cumulative_ratios_.size() - 1 &&
           r > cumulative_ratios_[idx]) {
      ++idx;
    }
    if (sources_[idx]->has_next()) {
      return sources_[idx]->next();
    }
  }

  // Fallback
  for (auto& s : sources_) {
    if (s->has_next()) {
      return s->next();
    }
  }

  throw std::runtime_error("MixingInstanceSource: all sources exhausted");
}

void MixingInstanceSource::reset() {
  rng_.seed(static_cast<unsigned>(seed_));
  for (auto& s : sources_) {
    s->reset();
  }
}

// ---------------------------------------------------------------------------
// ShuffledInstanceSource
// ---------------------------------------------------------------------------

ShuffledInstanceSource::ShuffledInstanceSource(
    std::unique_ptr<InstanceSource> source, int64_t buffer_size, int64_t seed)
    : source_(std::move(source)),
      buffer_size_(buffer_size),
      rng_(static_cast<unsigned>(seed)),
      seed_(seed) {}

bool ShuffledInstanceSource::has_next() const {
  if (filled_ && cursor_ < buffer_.size()) {
    return true;
  }
  return source_->has_next();
}

Instance ShuffledInstanceSource::next() {
  // If buffer is consumed or not yet filled, refill it
  if (!filled_ || cursor_ >= buffer_.size()) {
    buffer_.clear();
    cursor_ = 0;

    // Fill buffer from source
    while (static_cast<int64_t>(buffer_.size()) < buffer_size_ &&
           source_->has_next()) {
      buffer_.push_back(source_->next());
    }

    if (buffer_.empty()) {
      throw std::runtime_error("ShuffledInstanceSource: no more instances");
    }

    // Shuffle the buffer
    std::shuffle(buffer_.begin(), buffer_.end(), rng_);
    filled_ = true;
  }

  return buffer_[cursor_++];
}

void ShuffledInstanceSource::reset() {
  source_->reset();
  buffer_.clear();
  cursor_ = 0;
  filled_ = false;
  rng_.seed(static_cast<unsigned>(seed_));
}

}  // namespace olmo_cpp
