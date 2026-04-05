#include "olmo_cpp/model/fused_transformer.hpp"
#include <torch/nn/init.h>

namespace {
void trunc_normal_(torch::Tensor& t, double mean, double std, double a, double b, torch::Generator gen) {
  t.normal_(mean, std, gen);
  t.clamp_(a, b);
}
}  // namespace

namespace olmo_cpp {

FusedTransformerImpl::FusedTransformerImpl(const TransformerConfig& cfg)
    : blocks_(register_module("blocks", torch::nn::ModuleList())),
      lm_head_(register_module("lm_head", LMHead(cfg.d_model, cfg.vocab_size, true, cfg.layer_norm_eps))),
      embed_scale_(cfg.embed_scale),
      config_(cfg),
      use_multi_res_(cfg.use_multi_res),
      mtp_heads_(register_module("mtp_heads", torch::nn::ModuleList())) {

  // Choose embedding: multi-resolution or plain
  if (cfg.use_multi_res) {
    MultiResConfig mr_cfg;
    mr_cfg.char_trigram_buckets = cfg.multi_res_char_buckets;
    mr_cfg.phrase_buckets = cfg.multi_res_phrase_buckets;
    mr_cfg.role_embed_dim = cfg.multi_res_inner_dim;
    mr_cfg.char_embed_dim = cfg.multi_res_inner_dim;
    mr_cfg.phrase_embed_dim = cfg.multi_res_inner_dim;
    multi_res_embed_ = register_module("multi_res_embed",
        MultiResEmbedding(cfg.vocab_size, cfg.d_model, mr_cfg, cfg.bpe_vocab_path));
  } else {
    embeddings_ = register_module("embeddings", torch::nn::Embedding(cfg.vocab_size, cfg.d_model));
  }

  embedding_norm_ = RMSNorm(cfg.d_model, cfg.layer_norm_eps);
  register_module("embedding_norm", embedding_norm_.value());

  for (int64_t i = 0; i < cfg.n_layers; ++i) {
    blocks_->push_back(FusedTransformerBlock(cfg, i));
  }

  for (int64_t k = 0; k < cfg.num_mtp_heads; ++k) {
    mtp_heads_->push_back(MTPHead(cfg.d_model, cfg.layer_norm_eps));
  }
}

std::vector<RoPEBuffers> FusedTransformerImpl::get_rope_buffers(int64_t seq_len, torch::Device device,
                                                                torch::Dtype dtype) {
  if (seq_len <= cached_rope_len_ && !cached_rope_bufs_.empty() && dtype == cached_rope_dtype_) {
    return cached_rope_bufs_;
  }

  int64_t alloc_len = std::max(seq_len, cached_rope_len_ * 2);
  RotaryEmbedding rope(config_.get_head_dim(), config_.rope_theta);
  cached_rope_bufs_.resize(static_cast<size_t>(config_.n_layers));
  for (int64_t i = 0; i < config_.n_layers; ++i) {
    cached_rope_bufs_[i] = rope->get_buffers(alloc_len, device, dtype);
  }
  cached_rope_len_ = alloc_len;
  cached_rope_dtype_ = dtype;
  return cached_rope_bufs_;
}

void FusedTransformerImpl::init_weights(torch::optional<torch::Generator> gen) {
  torch::NoGradGuard no_grad;
  auto g = gen.value_or(torch::Generator());
  double emb_std = config_.embedding_init_std.value_or(config_.init_std);

  auto& emb_weight = use_multi_res_
      ? multi_res_embed_->semantic_weight()
      : embeddings_->weight;
  trunc_normal_(emb_weight, 0.0, emb_std, -3 * emb_std, 3 * emb_std, g);
  if (embed_scale_) {
    emb_weight.mul_(*embed_scale_);
  }

  for (int64_t i = 0; i < config_.n_layers; ++i) {
    auto block = blocks_->ptr<FusedTransformerBlockImpl>(i);
    for (auto& p : block->parameters()) {
      if (p.defined() && p.numel() > 0) {
        trunc_normal_(p, 0.0, config_.init_std, -3 * config_.init_std, 3 * config_.init_std, g);
      }
    }
  }

  double lm_std = 1.0 / std::sqrt(static_cast<double>(config_.d_model));
  trunc_normal_(lm_head_->w_out()->weight, 0.0, lm_std, -3 * lm_std, 3 * lm_std, g);

  for (int64_t k = 0; k < config_.num_mtp_heads; ++k) {
    auto head = mtp_heads_->ptr<MTPHeadImpl>(k);
    for (auto& p : head->parameters()) {
      if (p.defined() && p.numel() > 0) {
        trunc_normal_(p, 0.0, config_.init_std, -3 * config_.init_std, 3 * config_.init_std, g);
      }
    }
  }
}

torch::Tensor FusedTransformerImpl::forward_backbone(
    torch::Tensor input_ids,
    KVCache* kv_cache) {
  auto h = use_multi_res_
      ? multi_res_embed_->forward(input_ids)
      : embeddings_(input_ids);
  if (embed_scale_ && !use_multi_res_) {
    h = h * *embed_scale_;
  }
  h = (*embedding_norm_)(h);

  auto new_seq_len = input_ids.size(1);
  auto device = input_ids.device();

  int64_t cached_len = kv_cache ? kv_cache->seq_len() : 0;
  int64_t total_len = cached_len + new_seq_len;
  auto rope_bufs = get_rope_buffers(total_len, device, h.dtype().toScalarType());

  std::optional<int64_t> start_pos =
      kv_cache ? std::optional<int64_t>(cached_len) : std::nullopt;

  for (int64_t i = 0; i < config_.n_layers; ++i) {
    auto block = blocks_->ptr<FusedTransformerBlockImpl>(i);
    LayerKVCache* layer_cache = kv_cache ? &kv_cache->layers[static_cast<size_t>(i)] : nullptr;
    h = block->forward(h, &rope_bufs[i], start_pos, layer_cache);
  }

  return h;
}

torch::Tensor FusedTransformerImpl::forward(
    torch::Tensor input_ids,
    c10::optional<torch::Tensor> labels,
    int64_t ignore_index,
    KVCache* kv_cache) {

  auto h = forward_backbone(input_ids, kv_cache);
  auto logits = lm_head_(h);

  if (labels.has_value()) {
    auto main_loss = torch::nn::functional::cross_entropy(
        logits.view({-1, config_.vocab_size}),
        labels->view(-1),
        torch::nn::functional::CrossEntropyFuncOptions().ignore_index(ignore_index).reduction(torch::kMean));

    if (config_.num_mtp_heads > 0) {
      auto mtp_loss_sum = torch::zeros({}, logits.options());
      int64_t valid_heads = 0;

      for (int64_t k = 0; k < config_.num_mtp_heads; ++k) {
        int64_t shift = k + 1;
        int64_t seq_len = labels->size(1);

        if (shift >= seq_len) continue;

        auto h_trimmed = h.narrow(1, 0, seq_len - shift);
        auto labels_shifted = labels->narrow(1, shift, seq_len - shift);

        auto head = mtp_heads_->ptr<MTPHeadImpl>(k);
        auto mtp_h = head->forward(h_trimmed);
        auto mtp_logits = lm_head_(mtp_h);

        auto mtp_loss = torch::nn::functional::cross_entropy(
            mtp_logits.reshape({-1, config_.vocab_size}),
            labels_shifted.reshape(-1),
            torch::nn::functional::CrossEntropyFuncOptions().ignore_index(ignore_index).reduction(torch::kMean));

        mtp_loss_sum = mtp_loss_sum + mtp_loss;
        ++valid_heads;
      }

      if (valid_heads > 0) {
        main_loss = main_loss + config_.mtp_loss_weight * mtp_loss_sum / static_cast<double>(valid_heads);
      }
    }

    return main_loss;
  }
  return logits;
}

std::vector<torch::Tensor> FusedTransformerImpl::forward_mtp_draft(torch::Tensor hidden_state) {
  if (hidden_state.dim() == 1) {
    hidden_state = hidden_state.unsqueeze(0).unsqueeze(0);
  } else if (hidden_state.dim() == 2) {
    hidden_state = hidden_state.unsqueeze(0);
  }

  std::vector<torch::Tensor> draft_logits;
  draft_logits.reserve(static_cast<size_t>(config_.num_mtp_heads));

  for (int64_t k = 0; k < config_.num_mtp_heads; ++k) {
    auto head = mtp_heads_->ptr<MTPHeadImpl>(k);
    auto mtp_h = head->forward(hidden_state);
    auto logits = lm_head_(mtp_h);
    draft_logits.push_back(logits.squeeze(0).squeeze(0));
  }

  return draft_logits;
}

}  // namespace olmo_cpp
