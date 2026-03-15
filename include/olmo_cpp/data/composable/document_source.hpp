#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <functional>

namespace olmo_cpp {

/// A document is a sequence of token IDs with a boundary marker
struct Document {
  std::vector<int64_t> tokens;
  int64_t source_id = 0;  // which source this came from
};

/// Abstract document source
class DocumentSource {
 public:
  virtual ~DocumentSource() = default;
  virtual bool has_next() const = 0;
  virtual Document next() = 0;
  virtual void reset() = 0;
  virtual int64_t num_documents() const = 0;
};

/// In-memory document source
class InMemoryDocumentSource : public DocumentSource {
 public:
  explicit InMemoryDocumentSource(std::vector<Document> docs);
  bool has_next() const override;
  Document next() override;
  void reset() override;
  int64_t num_documents() const override;
 private:
  std::vector<Document> docs_;
  size_t cursor_ = 0;
};

/// Numpy document source: reads .npy files with document boundary markers
class NumpyDocumentSource : public DocumentSource {
 public:
  NumpyDocumentSource(const std::string& path, int64_t eos_token_id = -1);
  bool has_next() const override;
  Document next() override;
  void reset() override;
  int64_t num_documents() const override;
 private:
  std::vector<Document> docs_;
  size_t cursor_ = 0;
};

/// Concatenated document source: chains multiple sources
class ConcatenatedDocumentSource : public DocumentSource {
 public:
  explicit ConcatenatedDocumentSource(std::vector<std::unique_ptr<DocumentSource>> sources);
  bool has_next() const override;
  Document next() override;
  void reset() override;
  int64_t num_documents() const override;
 private:
  std::vector<std::unique_ptr<DocumentSource>> sources_;
  size_t current_source_ = 0;
};

/// Sampling document source: randomly samples from a source with replacement
class SamplingDocumentSource : public DocumentSource {
 public:
  SamplingDocumentSource(std::unique_ptr<DocumentSource> source, int64_t seed = 42);
  bool has_next() const override;
  Document next() override;
  void reset() override;
  int64_t num_documents() const override;
 private:
  std::vector<Document> all_docs_;
  std::mt19937 rng_;
  int64_t seed_;
};

/// Mixing document source: samples from multiple sources with given ratios
class MixingDocumentSource : public DocumentSource {
 public:
  MixingDocumentSource(std::vector<std::unique_ptr<DocumentSource>> sources,
                       std::vector<double> ratios, int64_t seed = 42);
  bool has_next() const override;
  Document next() override;
  void reset() override;
  int64_t num_documents() const override;
 private:
  std::vector<std::unique_ptr<DocumentSource>> sources_;
  std::vector<double> cumulative_ratios_;
  std::mt19937 rng_;
  int64_t seed_;
};

}  // namespace olmo_cpp
