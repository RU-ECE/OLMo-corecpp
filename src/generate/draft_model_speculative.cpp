/**
 * src/generate/draft_model_speculative.cpp
 *
 * Two-model speculative decoding (fast-inference [17]).
 *
 * Implementation parallels speculative_decode_step in chat.cpp but uses
 * a separate draft model instead of MTP heads:
 *   1. Draft model autoregressively generates k tokens (k forward passes
 *      on the draft model + sample at each).
 *   2. Target model verifies all k positions in ONE batched forward.
 *   3. Accept matching prefix.
 *
 * DRAFT — wiring into chat.cpp's main loop is a separate task. The
 * caller must supply both models already loaded onto the same device
 * with compatible tokenizers.
 */

#include "olmo_cpp/generate/draft_model_speculative.hpp"

#include <torch/torch.h>
#include <algorithm>

namespace olmo_cpp {

namespace {

// Greedy argmax of a 1D logits row. Same semantics as chat.cpp's
// rejection-test argmax in speculative verification.
int64_t argmax_1d(torch::Tensor logits_1d) {
  return logits_1d.argmax(-1).item<int64_t>();
}

}  // namespace

int64_t draft_model_speculative_step(
    DraftModelSpeculativeState& state,
    std::vector<int64_t>& all_tokens,
    torch::Device device,
    double /*temperature*/,
    int64_t /*top_k*/,
    double /*top_p*/,
    double /*repetition_penalty*/,
    std::mt19937& /*rng*/,
    BPETokenizer& tokenizer) {

  torch::NoGradGuard no_grad;
  int64_t eos_id = static_cast<int64_t>(tokenizer.eos_id());

  // Step 1: draft model produces draft_len tokens autoregressively.
  std::vector<int64_t> drafts;
  drafts.reserve(static_cast<size_t>(state.draft_len));

  int64_t cur = all_tokens.back();
  for (int64_t k = 0; k < state.draft_len; ++k) {
    auto inp = torch::tensor({cur}, torch::kInt64).unsqueeze(0).to(device);
    auto logits = (*state.draft_model)->forward(inp, c10::nullopt, -100, &state.draft_kv);
    auto next = logits.select(1, 0).squeeze(0);
    int64_t tok = argmax_1d(next);  // greedy draft for now
    drafts.push_back(tok);
    if (tok == eos_id) break;
    cur = tok;
  }
  state.total_drafted += static_cast<int64_t>(drafts.size());

  // Step 2: target model verifies [last_token, draft_0, ..., draft_{k-1}]
  //         in a single batched forward. The verify input length is
  //         1 + drafts.size() because we feed last_token to get the
  //         logits at position 0, then drafts to get logits at 1..k.
  std::vector<int64_t> verify_in;
  verify_in.reserve(1 + drafts.size());
  verify_in.push_back(all_tokens.back());
  for (auto d : drafts) verify_in.push_back(d);

  auto vinp = torch::tensor(at::IntArrayRef(verify_in.data(), verify_in.size()),
                            torch::kInt64).unsqueeze(0).to(device);

  auto target_snap = state.target_kv.snapshot();
  auto vlogits = (*state.target_model)->forward(vinp, c10::nullopt, -100, &state.target_kv);
  auto vlogits_cpu = vlogits.cpu().contiguous();

  // Step 3: walk verify positions, accept while target's argmax matches
  // the draft. On first mismatch, take target's choice and stop.
  int64_t accepted = 0;
  int64_t main_tok = argmax_1d(vlogits_cpu.select(0, 0).select(0, 0));
  all_tokens.push_back(main_tok);
  ++accepted;

  if (main_tok == eos_id) {
    // No further accepts possible.
    int64_t draft_kv_target = state.draft_kv.snapshot();
    (void)draft_kv_target;
    state.total_accepted += 0;
    return accepted;
  }

  int64_t drafts_accepted = 0;
  for (int64_t k = 0; k < static_cast<int64_t>(drafts.size()); ++k) {
    int64_t target_choice = argmax_1d(vlogits_cpu.select(0, 0).select(0, k + 1));
    if (target_choice == drafts[k]) {
      all_tokens.push_back(drafts[k]);
      ++accepted;
      ++drafts_accepted;
      if (drafts[k] == eos_id) break;
    } else {
      // Reject. Use target's choice; stop here.
      all_tokens.push_back(target_choice);
      ++accepted;
      break;
    }
  }
  state.total_accepted += drafts_accepted;

  // Rollback target KV to keep only the accepted positions.
  int64_t target_keep = target_snap + accepted;
  if (target_keep < state.target_kv.seq_len()) state.target_kv.rollback(target_keep);

  // Roll the draft KV back to where the accepted tokens end. Each drafted
  // token added one position to the draft_kv. We accepted (drafts_accepted)
  // of those, plus the main_tok which the draft model didn't see yet —
  // so the draft KV should be rolled back to drop only the *rejected*
  // drafts.
  int64_t draft_drop = static_cast<int64_t>(drafts.size()) - drafts_accepted;
  if (draft_drop > 0) {
    state.draft_kv.rollback(state.draft_kv.seq_len() - draft_drop);
  }

  return accepted;
}

}  // namespace olmo_cpp
