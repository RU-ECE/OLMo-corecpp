#pragma once
/**
 * include/olmo_cpp/generate/spec_core.hpp
 *
 * Exact speculative decoding with the model's own MTP heads, plus the shared
 * sampling helpers the draft-model path uses.
 *
 * One invariant, used by every speculative path here:
 *   - the KV cache holds every committed token EXCEPT the newest one, which
 *     is "pending" (committed, already returned to the caller, no KV yet);
 *   - h_last is the backbone output at the last cached token, so
 *     lm_head(h_last) is the distribution the pending token was drawn from,
 *     and MTP head k on h_last predicts the token k+1 places after it
 *     (training aligns head k with t+k+2, see TransformerImpl::forward).
 *
 * A step drafts d_0..d_{n-1} from h_last, runs ONE forward over
 * [pending, d_0..d_{n-1}], accepts the longest matching prefix, draws the
 * next pending token from the first rejected position (or the bonus position
 * when everything matches), and rolls the cache back to the accepted length.
 *
 * Greedy (temperature <= 0) output is token-for-token identical to plain
 * greedy decoding; tools/test_spec_greedy checks this. With temperature > 0,
 * drafts are sampled from the heads' distributions q and accepted with
 * min(1, p/q), rejections resample from norm(max(0, p - q)) (Leviathan et
 * al. 2023; Chen et al. 2023), so the output follows the model's own
 * distribution.
 *
 * This replaces the step that lived in tools/chat.cpp, which (bug 5 in the
 * speed plan) verified without a causal mask, kept one stale KV entry after
 * every step, duplicated the last prompt token at prefill, and ran two
 * forwards per step.
 */
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/transformer.hpp"

#include <torch/torch.h>
#include <random>
#include <vector>

namespace olmo_cpp {
namespace spec {

struct Sampling {
  double temperature = 0.0;         // <= 0: greedy
  int64_t top_k = 0;                // 0: off
  double top_p = 1.0;               // 1: off
  double repetition_penalty = 1.0;  // CTRL-style, once per distinct token in history
};

/// Logits [V] (any device/dtype) -> float64 CPU logits after the repetition
/// penalty over `history`.
torch::Tensor penalized_logits(const torch::Tensor& logits, const Sampling& s,
                               const std::vector<int64_t>& history);

/// Greedy choice under the penalty (what plain greedy decoding picks).
int64_t greedy_token(const torch::Tensor& logits, const Sampling& s,
                     const std::vector<int64_t>& history);

/// Sampling distribution [V] (float64, CPU, sums to 1) after penalty,
/// temperature, top-k and top-p. Requires s.temperature > 0.
torch::Tensor sampling_probs(const torch::Tensor& logits, const Sampling& s,
                             const std::vector<int64_t>& history);

/// Draw one token from a float64 CPU probability vector.
int64_t draw(const torch::Tensor& probs, std::mt19937& rng);

/// One position of speculative verification. p = target distribution, q =
/// the distribution `draft` was drawn from. Returns true and leaves *out
/// alone when the draft is accepted; otherwise returns false and writes the
/// replacement token (residual sample) to *out.
bool accept_or_resample(const torch::Tensor& p, const torch::Tensor& q, int64_t draft,
                        std::mt19937& rng, int64_t* out);

struct MtpState {
  MtpState(int64_t n_layers, torch::Device device) : kv(n_layers, device) {}
  KVCache kv;
  torch::Tensor h_last;   // [1, d_model]
  torch::Tensor logits_last;  // [V] = LMHead(h_last), reused for SL-MTP drafting
  int64_t pending = -1;   // newest committed token, not yet in kv
  bool done = false;      // an EOS was committed
  int64_t steps = 0, drafted = 0, accepted = 0;
};

/// Runs the prompt through the backbone, fills st.kv, sets h_last, and
/// draws the first generated token, which is appended to `history` and also
/// becomes st.pending. Returns that token.
int64_t mtp_prefill(Transformer& model, MtpState& st, std::vector<int64_t>& history,
                    const Sampling& s, std::mt19937& rng, int64_t eos_id);

/// One speculative step with up to `max_drafts` MTP drafts. Appends every
/// newly committed token to `history` and returns them (always >= 1 unless
/// st.done). `forced_drafts`, when non-null, replaces the MTP drafts (tests
/// use it to drive the accept / partial / reject paths with known tokens;
/// forced drafts count as greedy, q = one-hot).
std::vector<int64_t> mtp_step(Transformer& model, MtpState& st, std::vector<int64_t>& history,
                              int64_t max_drafts, const Sampling& s, std::mt19937& rng,
                              int64_t eos_id,
                              const std::vector<int64_t>* forced_drafts = nullptr);

/// Exact greedy TREE speculation (Medusa-style) on the same state. Depth d of
/// the tree holds the top-`fanout` candidates of MTP head d-1 (head 0 predicts
/// the token right after the pending one), under every node of depth d-1. One
/// forward verifies the whole tree against the cached context
/// (forward_tree_backbone: read-only prefix, node positions = cache length +
/// depth); the longest path whose tokens are the model's own greedy picks is
/// accepted, then one more forward over [pending, accepted...] writes their KV
/// and gives the next pending token. Output equals plain greedy decoding.
/// Greedy only (temperature <= 0). `forced_candidates[d]`, when given,
/// replaces head d's candidates (tests).
std::vector<int64_t> mtp_tree_step(Transformer& model, MtpState& st, std::vector<int64_t>& history,
                                   int64_t fanout, int64_t depth, const Sampling& s, int64_t eos_id,
                                   const std::vector<std::vector<int64_t>>* forced_candidates = nullptr);

/// Reference: plain one-token-at-a-time decoding with a KV cache, same
/// sampling rules. Used by tests and benchmarks as ground truth.
std::vector<int64_t> plain_generate(Transformer& model, const std::vector<int64_t>& prompt,
                                    int64_t n_new, const Sampling& s, std::mt19937& rng,
                                    int64_t eos_id);

}  // namespace spec
}  // namespace olmo_cpp
