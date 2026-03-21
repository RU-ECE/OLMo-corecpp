#pragma once
#include "olmo_cpp/data/composable/document_source.hpp"
#include <vector>
#include <memory>

namespace olmo_cpp {

/// Abstract token source: produces streams of tokens (no document boundaries)
class TokenSource {
 public:
  virtual ~TokenSource() = default;
  virtual bool has_next() const = 0;
  virtual int64_t next() = 0;
  virtual void reset() = 0;
  virtual int64_t total_tokens() const = 0;
};

/// Flattens documents into a token stream
class DocumentTokenSource : public TokenSource {
 public:
  explicit DocumentTokenSource(std::unique_ptr<DocumentSource> doc_source);
  bool has_next() const override;
  int64_t next() override;
  void reset() override;
  int64_t total_tokens() const override;
 private:
  std::unique_ptr<DocumentSource> doc_source_;
  Document current_doc_;
  size_t token_cursor_ = 0;
  bool doc_loaded_ = false;
  int64_t total_tokens_;
};

/// Sliced token source: takes a slice [start, end) of tokens
class SlicedTokenSource : public TokenSource {
 public:
  SlicedTokenSource(std::unique_ptr<TokenSource> source, int64_t start, int64_t end);
  bool has_next() const override;
  int64_t next() override;
  void reset() override;
  int64_t total_tokens() const override;
 private:
  std::unique_ptr<TokenSource> source_;
  int64_t start_, end_, pos_ = 0;
};

/// Mixing token source: interleaves tokens from multiple sources by ratio
class MixingTokenSource : public TokenSource {
 public:
  MixingTokenSource(std::vector<std::unique_ptr<TokenSource>> sources,
                    std::vector<double> ratios, int64_t seed = 42);
  bool has_next() const override;
  int64_t next() override;
  void reset() override;
  int64_t total_tokens() const override;
 private:
  std::vector<std::unique_ptr<TokenSource>> sources_;
  std::vector<double> cumulative_ratios_;
  std::mt19937 rng_;
  int64_t seed_;
};

}  // namespace olmo_cpp
