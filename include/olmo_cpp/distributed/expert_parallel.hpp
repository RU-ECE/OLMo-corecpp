#pragma once

#include <torch/torch.h>
#include <optional>
#include <tuple>

#ifdef OLMO_HAS_DDP
#include <torch/csrc/distributed/c10d/Backend.hpp>
#endif

namespace olmo_cpp {

/// Expert parallelism: distributes MoE experts across ranks.
/// Uses all-to-all to dispatch tokens to the rank owning their assigned expert.
class ExpertParallelContext {
 public:
#ifdef OLMO_HAS_DDP
  static std::optional<ExpertParallelContext> create(
      c10::intrusive_ptr<c10d::Backend> backend, int num_experts, int ep_size);
#endif

  /// Dispatch tokens to expert-owning ranks via all-to-all.
  /// tokens: [num_tokens, D], expert_ids: [num_tokens, top_k]
  /// Returns: local_tokens [local_count, D], metadata for combine
  struct DispatchResult {
    torch::Tensor local_tokens;  // tokens assigned to this rank's experts
    torch::Tensor local_ids;     // local expert indices (0..experts_per_rank-1)
    torch::Tensor weights;       // routing weights for local tokens
    torch::Tensor send_counts;   // how many tokens sent to each rank
    torch::Tensor recv_counts;   // how many tokens received from each rank
  };

  DispatchResult dispatch(torch::Tensor tokens, torch::Tensor expert_ids,
                          torch::Tensor weights);

  /// Combine: gather expert outputs back to original ranks via all-to-all
  torch::Tensor combine(torch::Tensor expert_outputs,
                         const DispatchResult& dispatch_info);

  int local_expert_start() const { return rank_ * experts_per_rank_; }
  int local_expert_end() const { return (rank_ + 1) * experts_per_rank_; }
  int num_local_experts() const { return experts_per_rank_; }
  int rank() const { return rank_; }
  int ep_size() const { return ep_size_; }

 private:
#ifdef OLMO_HAS_DDP
  ExpertParallelContext(c10::intrusive_ptr<c10d::Backend> backend,
                        int rank, int ep_size, int num_experts);
  c10::intrusive_ptr<c10d::Backend> backend_;
#else
  ExpertParallelContext(int rank, int ep_size, int num_experts);
#endif
  int rank_, ep_size_, num_experts_, experts_per_rank_;
};

}  // namespace olmo_cpp
