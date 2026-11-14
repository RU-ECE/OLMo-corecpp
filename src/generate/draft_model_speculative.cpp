/**
 * src/generate/draft_model_speculative.cpp
 *
 * Two-model speculative decoding (fast-inference [17]).
 *
 * Invariant (shared with spec_core.hpp): both KV caches hold every
 * committed token except state.pending, the newest one. A step:
 *   1. The draft model runs pending, d_0, ..., d_{n-2} and proposes
 *      d_0 .. d_{n-1} (n draft forwards, n cache entries).
 *   2. The target runs [pending, d_0 .. d_{n-1}] in ONE forward. Row i is
 *      its distribution for the slot d_i was proposed for; row n is the
 *      bonus slot.
 *   3. Accept the longest prefix where the target agrees (greedy) or passes
 *      min(1, p/q) (sampling); the first rejected slot is resampled from
 *      norm(max(0, p - q)) and becomes the new pending token.
 *   4. Both caches roll back to old + 1 + j entries (pending + j accepted
 *      drafts). When every draft is accepted the draft model also runs
 *      d_{n-1}, because it never saw it.
 *
 * The previous version compared d_i with the target's row i+1 (the
 * prediction AFTER d_i), so accepted text was not the target's text, and
 * its draft-cache drop was off by one (speed-plan bug 5).
 */

#include "olmo_cpp/generate/draft_model_speculative.hpp"
#include "olmo_cpp/generate/spec_core.hpp"

#include <torch/torch.h>
#include <algorithm>
#include <vector>

namespace olmo_cpp {

namespace {

torch::Tensor ids_tensor(const std::vector<int64_t>& ids, torch::Device device) {
  return torch::tensor(at::IntArrayRef(ids.data(), ids.size()), torch::kInt64)
      .unsqueeze(0)
      .to(device);
}

spec::Sampling make_sampling(double temperature, int64_t top_k, double top_p, double rep) {
  spec::Sampling s;
  s.temperature = temperature;
  s.top_k = top_k;
  s.top_p = top_p;
  s.repetition_penalty = rep;
  return s;
}

}  // namespace

int64_t draft_model_speculative_prefill(
    DraftModelSpeculativeState& state,
    std::vector<int64_t>& all_tokens,
    torch::Device device,
    double temperature,
    int64_t top_k,
    double top_p,
    double repetition_penalty,
    std::mt19937& rng,
    BPETokenizer& tokenizer) {
  TORCH_CHECK(!all_tokens.empty(), "draft_model_speculative_prefill: empty prompt");
  torch::NoGradGuard no_grad;
  const auto s = make_sampling(temperature, top_k, top_p, repetition_penalty);
  state.target_kv.clear();
  state.draft_kv.clear();
  auto ids = ids_tensor(all_tokens, device);
  auto logits = (*state.target_model)->forward(ids, c10::nullopt, -100, &state.target_kv);
  (*state.draft_model)->forward(ids, c10::nullopt, -100, &state.draft_kv);
  auto last = logits.select(1, logits.size(1) - 1);
  const int64_t tok = s.temperature <= 0.0 ? spec::greedy_token(last, s, all_tokens)
                                           : spec::draw(spec::sampling_probs(last, s, all_tokens), rng);
  all_tokens.push_back(tok);
  state.pending = tok;
  state.done = (tok == static_cast<int64_t>(tokenizer.eos_id()));
  return 1;
}

int64_t draft_model_speculative_step(
    DraftModelSpeculativeState& state,
    std::vector<int64_t>& all_tokens,
    torch::Device device,
    double temperature,
    int64_t top_k,
    double top_p,
    double repetition_penalty,
    std::mt19937& rng,
    BPETokenizer& tokenizer) {
  if (state.done) return 0;
  TORCH_CHECK(state.pending >= 0, "draft_model_speculative_step: call draft_model_speculative_prefill first");
  torch::NoGradGuard no_grad;
  const int64_t eos_id = static_cast<int64_t>(tokenizer.eos_id());
  const auto s = make_sampling(temperature, top_k, top_p, repetition_penalty);
  const bool greedy = s.temperature <= 0.0;

  const int64_t target_snap = state.target_kv.seq_len();
  const int64_t draft_snap = state.draft_kv.seq_len();

  // 1. Draft n tokens, starting from the pending token.
  std::vector<int64_t> drafts;
  std::vector<torch::Tensor> q;
  auto hist = all_tokens;  // ends with pending
  auto inp = torch::empty({1, 1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  int64_t cur = state.pending;
  for (int64_t k = 0; k < state.draft_len; ++k) {
    inp.fill_(cur);
    auto row = (*state.draft_model)->forward(inp, c10::nullopt, -100, &state.draft_kv).select(1, 0);
    int64_t d;
    if (greedy) {
      d = spec::greedy_token(row, s, hist);
    } else {
      q.push_back(spec::sampling_probs(row, s, hist));
      d = spec::draw(q.back(), rng);
    }
    drafts.push_back(d);
    hist.push_back(d);
    cur = d;
    if (d == eos_id) break;
  }
  const int64_t n = static_cast<int64_t>(drafts.size());
  state.total_drafted += n;

  // 2. Verify [pending, d_0 .. d_{n-1}] in one target forward.
  std::vector<int64_t> verify = {state.pending};
  verify.insert(verify.end(), drafts.begin(), drafts.end());
  auto logits = (*state.target_model)
                    ->forward(ids_tensor(verify, device), c10::nullopt, -100, &state.target_kv)
                    .select(0, 0);  // [n+1, V]

  // 3. Accept the longest agreeing prefix.
  std::vector<int64_t> out;
  hist = all_tokens;
  int64_t j = 0, next = -1;
  bool stopped_on_eos = false;
  for (; j < n; ++j) {
    const auto row = logits.select(0, j);
    const int64_t d = drafts[static_cast<size_t>(j)];
    if (greedy) {
      const int64_t c = spec::greedy_token(row, s, hist);
      if (c != d) { next = c; break; }
    } else {
      int64_t x = -1;
      if (!spec::accept_or_resample(spec::sampling_probs(row, s, hist), q[static_cast<size_t>(j)], d, rng, &x)) {
        next = x;
        break;
      }
    }
    out.push_back(d);
    hist.push_back(d);
    if (d == eos_id) { stopped_on_eos = true; ++j; break; }
  }
  state.total_accepted += j;

  if (stopped_on_eos) {
    state.done = true;
  } else {
    if (next < 0) {  // every draft accepted: the bonus slot
      const auto row = logits.select(0, n);
      next = greedy ? spec::greedy_token(row, s, hist)
                    : spec::draw(spec::sampling_probs(row, s, hist), rng);
    }
    out.push_back(next);
    state.pending = next;
    state.done = (next == eos_id);

    // 4. Both caches: old + pending + j accepted drafts.
    state.target_kv.rollback(target_snap + 1 + j);
    const int64_t draft_have = state.draft_kv.seq_len() - draft_snap;  // == n
    if (draft_have > j + 1) {
      state.draft_kv.rollback(draft_snap + 1 + j);
    } else if (draft_have == j) {
      // The draft never ran verify[j] (the last accepted draft, or pending
      // when n == 0). Run it so the draft cache matches the target's.
      inp.fill_(verify[static_cast<size_t>(j)]);
      (*state.draft_model)->forward(inp, c10::nullopt, -100, &state.draft_kv);
    }
  }
  all_tokens.insert(all_tokens.end(), out.begin(), out.end());
  return static_cast<int64_t>(out.size());
}

}  // namespace olmo_cpp
