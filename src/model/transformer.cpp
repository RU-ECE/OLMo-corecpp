#include "olmo_cpp/model/transformer.hpp"
#include <torch/nn/init.h>

namespace {
void trunc_normal_(torch::Tensor& t, double mean, double std, double a, double b, torch::Generator gen) {
  t.normal_(mean, std, gen);
  t.clamp_(a, b);
}
}  // namespace

namespace olmo_cpp {

TransformerImpl::TransformerImpl(const TransformerConfig& cfg)
    : embeddings_(register_module("embeddings", torch::nn::Embedding(cfg.vocab_size, cfg.d_model))),
      blocks_(register_module("blocks", torch::nn::ModuleList())),
      lm_head_(register_module("lm_head", LMHead(cfg.d_model, cfg.vocab_size, true, cfg.layer_norm_eps))),
      embed_scale_(cfg.embed_scale),
      config_(cfg) {
  embedding_norm_ = RMSNorm(cfg.d_model, cfg.layer_norm_eps);
  register_module("embedding_norm", embedding_norm_.value());

  for (int64_t i = 0; i < cfg.n_layers; ++i) {
    blocks_->push_back(ReorderedNormTransformerBlock(cfg, i));
  }
}

std::vector<RoPEBuffers> TransformerImpl::get_rope_buffers(int64_t seq_len, torch::Device device) {
  if (seq_len <= cached_rope_len_ && !cached_rope_bufs_.empty()) {
    return cached_rope_bufs_;
  }

  int64_t alloc_len = std::max(seq_len, cached_rope_len_ * 2);
  RotaryEmbedding rope(config_.get_head_dim(), config_.rope_theta);
  cached_rope_bufs_.resize(static_cast<size_t>(config_.n_layers));
  for (int64_t i = 0; i < config_.n_layers; ++i) {
    cached_rope_bufs_[i] = rope->get_buffers(alloc_len, device);
  }
  cached_rope_len_ = alloc_len;
  return cached_rope_bufs_;
}

void TransformerImpl::init_weights(torch::optional<torch::Generator> gen) {
  torch::NoGradGuard no_grad;
  auto g = gen.value_or(torch::Generator());
  double emb_std = config_.embedding_init_std.value_or(config_.init_std);
  trunc_normal_(embeddings_->weight, 0.0, emb_std, -3 * emb_std, 3 * emb_std, g);
  if (embed_scale_) {
    embeddings_->weight.mul_(*embed_scale_);
  }

  for (int64_t i = 0; i < config_.n_layers; ++i) {
    auto block = blocks_->ptr<ReorderedNormTransformerBlockImpl>(i);
    for (auto& p : block->parameters()) {
      if (p.defined() && p.numel() > 0) {
        trunc_normal_(p, 0.0, config_.init_std, -3 * config_.init_std, 3 * config_.init_std, g);
      }
    }
  }

  double lm_std = 1.0 / std::sqrt(static_cast<double>(config_.d_model));
  trunc_normal_(lm_head_->w_out()->weight, 0.0, lm_std, -3 * lm_std, 3 * lm_std, g);
}

torch::Tensor TransformerImpl::forward(
    torch::Tensor input_ids,
    c10::optional<torch::Tensor> labels,
    int64_t ignore_index,
    KVCache* kv_cache) {

  auto h = embeddings_(input_ids);
  if (embed_scale_) {
    h = h * *embed_scale_;
  }
  h = (*embedding_norm_)(h);

  auto new_seq_len = input_ids.size(1);
  auto device = input_ids.device();

  int64_t cached_len = kv_cache ? kv_cache->seq_len() : 0;
  int64_t total_len = cached_len + new_seq_len;
  auto rope_bufs = get_rope_buffers(total_len, device);

  std::optional<int64_t> start_pos =
      kv_cache ? std::optional<int64_t>(cached_len) : std::nullopt;

  for (int64_t i = 0; i < config_.n_layers; ++i) {
    auto block = blocks_->ptr<ReorderedNormTransformerBlockImpl>(i);
    LayerKVCache* layer_cache = kv_cache ? &kv_cache->layers[static_cast<size_t>(i)] : nullptr;
    h = block->forward(h, &rope_bufs[i], start_pos, layer_cache);
  }

  auto logits = lm_head_(h);

  if (labels.has_value()) {
    return torch::nn::functional::cross_entropy(
        logits.view({-1, config_.vocab_size}),
        labels->view(-1),
        torch::nn::functional::CrossEntropyFuncOptions().ignore_index(ignore_index).reduction(torch::kMean));
  }
  return logits;
}

}  // namespace olmo_cpp
