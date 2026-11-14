/**
 * include/olmo_cpp/generate/draft_model_speculative.hpp
 *
 * Two-model speculative decoding — fast-inference [17].
 *
 * Standard speculative decoding with a small draft model and a larger
 * target model:
 *
 *   1. Draft model produces k tokens autoregressively (cheap).
 *   2. Target model verifies all k positions in one forward pass.
 *   3. Accept the longest matching prefix; reject and resample at
 *      first mismatch.
 *
 * Different from MTP speculative (which uses heads of the same model):
 * the draft is a smaller separate model. Typically yields HIGHER
 * acceptance rate than MTP because the draft model has its own
 * representation, but adds a separate model load + per-step forward.
 *
 * Wired into tools/chat.cpp (--draft-checkpoint). The step was rewritten
 * for speed-plan bug 5: drafts used to be compared with the target's
 * prediction one position too late, and both KV caches kept a stale entry.
 */

#pragma once

#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"

#include <torch/torch.h>
#include <random>
#include <vector>

namespace olmo_cpp {

/// State kept across speculative steps when using a separate draft model.
/// Both models use the same tokenizer (so token ids are interchangeable);
/// each has its own KV cache.
struct DraftModelSpeculativeState {
  Transformer* target_model;
  Transformer* draft_model;
  KVCache target_kv;
  KVCache draft_kv;
  int64_t draft_len;       // dynamic k, may shift via accept-rate
  int64_t total_drafted;
  int64_t total_accepted;
  // Same invariant as spec_core.hpp: both caches hold every committed token
  // except `pending`, the newest one. Set by draft_model_speculative_prefill.
  int64_t pending = -1;
  bool done = false;

  DraftModelSpeculativeState(Transformer* tgt, Transformer* drf,
                             int64_t target_n_layers, int64_t draft_n_layers,
                             torch::Device device)
      : target_model(tgt), draft_model(drf),
        target_kv(target_n_layers, device),
        draft_kv(draft_n_layers, device),
        draft_len(4),
        total_drafted(0), total_accepted(0) {}
};

/// Runs the prompt (all_tokens) through both models, draws the first new
/// token from the target, appends it to all_tokens and makes it pending.
/// Must be called once before the first step. Returns 1.
int64_t draft_model_speculative_prefill(
    DraftModelSpeculativeState& state,
    std::vector<int64_t>& all_tokens,
    torch::Device device,
    double temperature,
    int64_t top_k,
    double top_p,
    double repetition_penalty,
    std::mt19937& rng,
    BPETokenizer& tokenizer);

/// One speculative step. Returns the number of tokens appended to
/// all_tokens (>= 1 unless an EOS was already committed). Greedy output is
/// identical to greedy decoding with the target alone (tools/test_spec_greedy).
int64_t draft_model_speculative_step(
    DraftModelSpeculativeState& state,
    std::vector<int64_t>& all_tokens,
    torch::Device device,
    double temperature,
    int64_t top_k,
    double top_p,
    double repetition_penalty,
    std::mt19937& rng,
    BPETokenizer& tokenizer);

}  // namespace olmo_cpp
