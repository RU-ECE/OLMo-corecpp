#pragma once

#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/fused_transformer.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include <torch/torch.h>
#include <vector>
#include <functional>

namespace olmo_cpp {

/// Speculative decoding using Multi-Token Prediction (MTP) heads.
///
/// The key insight: MTP heads are trained to predict tokens at positions
/// t+1, t+2, ..., t+K alongside the main next-token prediction. During
/// inference, we use these as a "draft" model to propose K candidate tokens,
/// then verify all K in a single forward pass of the main model.
///
/// Speedup: Instead of K sequential forward passes (one per token),
/// we do 1 MTP draft (cheap, reuses cached hidden states) + 1 verification
/// forward pass. If acceptance rate is high (typically 60-80% for well-trained
/// MTP heads), we generate 2-4x faster.
///
/// For the paper: This is "free" speculative decoding — no separate draft model
/// needed. The MTP heads add <5% training cost but enable 2-4x inference speedup.

struct SpeculativeConfig {
  int64_t max_draft_tokens = 0;   // 0 = use all MTP heads
  double temperature = 1.0;
  int64_t top_k = 50;
  double top_p = 0.95;
  bool greedy = false;            // true = argmax sampling
};

struct SpeculativeResult {
  std::vector<int64_t> tokens;
  int64_t num_draft_tokens = 0;   // how many we proposed
  int64_t num_accepted = 0;       // how many were accepted
  double acceptance_rate = 0.0;
  int64_t total_forward_passes = 0;
  double tokens_per_forward = 0.0; // key metric: higher = better
};

/// Sample a token from logits with temperature, top-k, top-p
int64_t sample_token(torch::Tensor logits, double temperature = 1.0,
                     int64_t top_k = 50, double top_p = 0.95, bool greedy = false);

/// Speculative decode one "step" (may produce 1..K+1 tokens).
///
/// 1. Get hidden state from last generated position
/// 2. Use MTP heads to draft K candidates
/// 3. Run verification forward pass on all candidates
/// 4. Accept prefix that matches, resample the first rejection
///
/// Works with both Transformer and FusedTransformer via template.
template <typename ModelType>
SpeculativeResult speculative_decode_step(
    ModelType& model,
    KVCache& kv_cache,
    int64_t last_token,
    torch::Device device,
    const SpeculativeConfig& config = {});

/// Generate a full sequence using speculative decoding.
template <typename ModelType>
SpeculativeResult speculative_generate(
    ModelType& model,
    torch::Tensor prompt_ids,   // [1, prompt_len]
    int64_t max_new_tokens,
    torch::Device device,
    const SpeculativeConfig& config = {},
    std::function<bool(int64_t)> stop_fn = nullptr);  // return true to stop

}  // namespace olmo_cpp
