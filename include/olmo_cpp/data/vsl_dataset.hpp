#pragma once
#include "olmo_cpp/data/composable/instance_source.hpp"
#include "olmo_cpp/data/composable/document_source.hpp"
#include <memory>

namespace olmo_cpp {

/// Variable Sequence Length curriculum strategies
enum class VSLCurriculum { Natural, Linear, Quadratic };

/// VSL instance source: dynamically adjusts sequence length during training
class VSLInstanceSource : public InstanceSource {
 public:
  VSLInstanceSource(std::unique_ptr<DocumentSource> source,
                    int64_t min_seq_len, int64_t max_seq_len,
                    int64_t warmup_steps, VSLCurriculum curriculum = VSLCurriculum::Linear,
                    int64_t pad_token_id = 0);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
  void set_step(int64_t step);
  int64_t current_seq_len() const { return current_seq_len_; }
 private:
  int64_t compute_seq_len(int64_t step) const;
  std::unique_ptr<DocumentSource> source_;
  int64_t min_seq_len_, max_seq_len_, warmup_steps_;
  VSLCurriculum curriculum_;
  int64_t pad_token_id_;
  int64_t current_step_ = 0;
  int64_t current_seq_len_;
  std::vector<int64_t> buffer_;
};

}  // namespace olmo_cpp
