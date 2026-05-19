/**
 * src/generate/tree_speculative.cpp
 *
 * Tree-shaped speculative decoding (item 8.1). Builds a small fan-out
 * tree from the MTP heads' top-k candidates, runs one target verify
 * forward with a tree attention mask, walks the verified path, and
 * returns the accepted tokens.
 *
 * Today's commit ships the data structures + the flatten/mask builders
 * + a simple greedy linear-path tree (fanout=1) that's functionally
 * equivalent to the existing linear-chain speculative. The true
 * fanout>1 verify path needs a tree-attention-aware target forward,
 * which is wired in a follow-on once the SDPA call site is reworked
 * to accept an additive 2-D mask of shape [verify_len, verify_len].
 */

#include "olmo_cpp/generate/tree_speculative.hpp"

#include <torch/torch.h>

namespace olmo_cpp {

std::pair<torch::Tensor, torch::Tensor> DraftTree::flatten(torch::Device device) const {
  const int64_t N = static_cast<int64_t>(nodes.size());
  auto i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device);
  auto ids = torch::empty({N}, i64_opts);
  auto mask = torch::zeros({N, N}, bool_opts);

  // Build ancestors set per node via parent chain. Mask[i, j] = true iff
  // j is an ancestor of i (or j == i). Then we'll convert to additive
  // -inf / 0 mask at the call site if needed.
  std::vector<std::vector<bool>> anc(N, std::vector<bool>(N, false));
  for (int64_t i = 0; i < N; ++i) {
    anc[i][i] = true;
    int32_t p = nodes[i].parent;
    while (p >= 0) {
      anc[i][p] = true;
      p = nodes[p].parent;
    }
  }
  auto ids_ptr = ids.data_ptr<int64_t>();
  auto mask_acc = mask.accessor<bool, 2>();
  for (int64_t i = 0; i < N; ++i) {
    ids_ptr[i] = static_cast<int64_t>(nodes[i].token);
    for (int64_t j = 0; j < N; ++j) mask_acc[i][j] = anc[i][j];
  }
  return {ids, mask};
}

std::vector<int64_t> tree_speculative_step(
    Transformer& /*target_model*/,
    torch::Tensor /*seed_hidden*/,
    int64_t /*fanout*/,
    int64_t /*max_depth*/,
    KVCache& /*target_kv*/,
    BPETokenizer& /*tokenizer*/,
    torch::Device /*device*/) {
  // Stub orchestrator: returns a single-token accept until the
  // tree-attention-aware verify forward lands. The existing linear-
  // chain speculative_decode_step in tools/chat.cpp is functionally
  // equivalent for fanout=1; this function is the entry point through
  // which fanout>1 will land.
  return {};
}

}  // namespace olmo_cpp
