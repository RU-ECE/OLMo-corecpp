#include "zwt/layers/transformer.hpp"

#include <stdexcept>

namespace zwt {

namespace {

TransformerBlock::Config make_block_cfg(const Transformer::Config& c) {
  TransformerBlock::Config b;
  b.d_model   = c.d_model;
  b.n_heads   = c.n_heads;
  b.head_dim  = c.head_dim;
  b.d_ffn     = c.d_ffn;
  b.max_seq   = c.max_seq;
  b.rope_base = c.rope_base;
  b.norm_eps  = c.norm_eps;
  b.bias      = c.bias;
  return b;
}

}  // namespace

Transformer::Transformer(const Config& cfg, DType dtype, Device device,
                         uint64_t init_seed)
    : cfg_(cfg),
      tok_emb_(cfg.vocab_size, cfg.d_model, dtype, device),
      final_norm_(cfg.d_model, cfg.norm_eps, dtype, device),
      lm_head_(cfg.d_model, cfg.vocab_size, /*bias=*/false, dtype, device,
               init_seed ^ 0xBABEDEEDULL) {
  if (cfg.vocab_size <= 0 || cfg.d_model <= 0 || cfg.n_layers <= 0 ||
      cfg.n_heads <= 0 || cfg.head_dim <= 0 || cfg.d_ffn <= 0 || cfg.max_seq <= 0) {
    throw std::runtime_error("Transformer: invalid config");
  }
  if (cfg.d_model != cfg.n_heads * cfg.head_dim) {
    throw std::runtime_error("Transformer: d_model must equal n_heads * head_dim");
  }

  blocks_.reserve(static_cast<size_t>(cfg.n_layers));
  for (int64_t i = 0; i < cfg.n_layers; ++i) {
    uint64_t seed = init_seed + static_cast<uint64_t>(i) * 0x9E37'79B1ULL;
    blocks_.emplace_back(std::make_unique<TransformerBlock>(
        make_block_cfg(cfg), dtype, device, seed));
  }

  if (cfg.tie_embeddings) {
    // Share embedding weights with the LM head. The lm_head_.weight_ buffer
    // is replaced with a view over tok_emb_'s weight. Gradients must then be
    // accumulated into a single parameter during backward.
    throw std::runtime_error("Transformer: weight tying not yet implemented");
  }
}

Tensor Transformer::forward(const Tensor& tokens) {
  Tensor h = tok_emb_.forward(tokens);          // [B, S, d_model]
  for (auto& blk : blocks_) {
    h = blk->forward(h);
  }
  h = final_norm_.forward(h);
  return lm_head_.forward(h);                   // [B, S, vocab]
}

Tensor Transformer::backward(const Tensor& grad_logits) {
  Tensor g = lm_head_.backward(grad_logits);
  g = final_norm_.backward(g);
  // Walk blocks in reverse.
  for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
    g = (*it)->backward(g);
  }
  return tok_emb_.backward(g);
}

void Transformer::collect_params(std::vector<Parameter*>& out) {
  tok_emb_.collect_params(out);
  for (auto& blk : blocks_) blk->collect_params(out);
  final_norm_.collect_params(out);
  lm_head_.collect_params(out);
}

}  // namespace zwt
