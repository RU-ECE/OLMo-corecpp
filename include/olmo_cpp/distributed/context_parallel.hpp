#pragma once

#include <torch/torch.h>
#include <optional>

#ifdef OLMO_HAS_DDP
#include <torch/csrc/distributed/c10d/Backend.hpp>
#endif

namespace olmo_cpp {

/// Context parallelism: splits sequence dimension across ranks.
/// Uses ring attention to compute full causal attention over distributed sequence chunks.
class ContextParallelContext {
 public:
#ifdef OLMO_HAS_DDP
  static std::optional<ContextParallelContext> create(
      c10::intrusive_ptr<c10d::Backend> backend, int cp_size);
#endif

  /// Scatter: split sequence into per-rank chunks [B,S,D] -> [B,S/cp,D]
  torch::Tensor scatter_sequence(torch::Tensor x);

  /// Ring attention: compute full causal attention across distributed chunks
  /// q,k,v: [B,H,S/cp,D] -> output: [B,H,S/cp,D]
  torch::Tensor ring_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);

  /// Gather: reassemble full sequence [B,S/cp,D] -> [B,S,D]
  torch::Tensor gather_sequence(torch::Tensor x);

  int cp_rank() const { return rank_; }
  int cp_size() const { return cp_size_; }

 private:
#ifdef OLMO_HAS_DDP
  ContextParallelContext(c10::intrusive_ptr<c10d::Backend> backend, int rank, int cp_size);
  c10::intrusive_ptr<c10d::Backend> backend_;
#else
  ContextParallelContext(int rank, int cp_size);
#endif
  int rank_;
  int cp_size_;
};

}  // namespace olmo_cpp
