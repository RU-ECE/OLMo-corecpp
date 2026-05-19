#pragma once

/**
 * include/olmo_cpp/generate/tree_speculative.hpp
 *
 * Tree-shaped speculative decoding (item 8.1) — Medusa / EAGLE style.
 *
 * Standard (linear-chain) speculative decoding from chat.cpp drafts k
 * tokens sequentially and verifies them in one target forward. Tree
 * speculative drafts a TREE of candidates (each MTP head can fan out
 * multiple branches), verifies the entire tree in ONE target forward
 * with a "tree attention mask" that ensures each branch only sees its
 * own ancestors, and accepts the longest matching path.
 *
 * Expected speedup over linear-chain spec: 1.4-1.8× on top of the
 * 2-2.5× linear chain already gives. Comes from a higher expected
 * acceptance-rate per token-position (you have multiple alternatives
 * at each step).
 *
 * Surface here:
 *   - DraftTree builder: stack candidate tokens + their parent indices
 *     into a flat representation suitable for one batched verify pass.
 *   - tree_attention_mask: produces the [n_nodes, n_nodes] additive
 *     mask that the target SDPA call uses.
 *   - tree_speculative_step(...): the orchestration routine. Returns
 *     the accepted path's tokens.
 */

#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"

#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

struct DraftTreeNode {
  int32_t token;
  int32_t parent;  // index of parent in the tree node array; -1 = root
  int32_t depth;
};

struct DraftTree {
  std::vector<DraftTreeNode> nodes;
  /// Build a flat [n_nodes] input id sequence and the [n_nodes, n_nodes]
  /// causal attention mask that allows each node to attend only to its
  /// ancestors (including itself).
  std::pair<torch::Tensor, torch::Tensor> flatten(torch::Device device) const;
};

/// Tree-spec step: drafts a width-fanout-`fanout` tree of depth
/// `max_depth` using the MTP heads, runs ONE verify forward on the
/// target, accepts the longest matching path. Returns the path's
/// tokens.
std::vector<int64_t> tree_speculative_step(
    Transformer& target_model,
    torch::Tensor seed_hidden,
    int64_t fanout,
    int64_t max_depth,
    KVCache& target_kv,
    BPETokenizer& tokenizer,
    torch::Device device);

}  // namespace olmo_cpp
