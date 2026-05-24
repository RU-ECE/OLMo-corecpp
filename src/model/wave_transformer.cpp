#include "olmo_cpp/model/wave_transformer.hpp"
#include "olmo_cpp/model/block.hpp"
#include "olmo_cpp/model/lm_head.hpp"
#include "olmo_cpp/model/wave_block.hpp"
#include <torch/torch.h>
#include <stdexcept>

namespace olmo_cpp {

using MixerType = TransformerConfig::SequenceMixerType;

WaveTransformerImpl::WaveTransformerImpl(const TransformerConfig& cfg)
    : config_(cfg),
      embeddings_(register_module(
          "embeddings",
          torch::nn::Embedding(cfg.vocab_size, cfg.d_model))),
      blocks_(register_module("blocks", torch::nn::ModuleList())),
      lm_head_(register_module(
          "lm_head",
          LMHead(cfg.d_model, cfg.vocab_size, /*use_norm=*/true, cfg.layer_norm_eps)))
{
  if (cfg.layer_norm_eps > 0) {
    embedding_norm_ = register_module(
        "embedding_norm",
        RMSNorm(cfg.d_model, cfg.layer_norm_eps));
  }

  const MixerType mix = cfg.sequence_mixer;
  is_wave_layer_.resize(cfg.n_layers);

  for (int64_t i = 0; i < cfg.n_layers; ++i) {
    const bool even = (i % 2 == 0);

    if (mix == MixerType::Wave) {
      is_wave_layer_[i] = true;
      blocks_->push_back(WaveBlock(cfg, i, /*use_wave_mixer=*/true));

    } else if (mix == MixerType::Hyena) {
      is_wave_layer_[i] = false;
      blocks_->push_back(WaveBlock(cfg, i, /*use_wave_mixer=*/false));

    } else if (mix == MixerType::HybridWave) {
      if (even) {
        is_wave_layer_[i] = true;
        blocks_->push_back(WaveBlock(cfg, i, /*use_wave_mixer=*/true));
      } else {
        is_wave_layer_[i] = false;
        blocks_->push_back(ReorderedNormTransformerBlock(cfg, i));
      }

    } else if (mix == MixerType::HybridHyena) {
      if (even) {
        is_wave_layer_[i] = false;
        blocks_->push_back(WaveBlock(cfg, i, /*use_wave_mixer=*/false));
      } else {
        is_wave_layer_[i] = false;
        blocks_->push_back(ReorderedNormTransformerBlock(cfg, i));
      }

    } else {
      throw std::runtime_error(
          "WaveTransformer: sequence_mixer must be Wave, Hyena, "
          "HybridWave, or HybridHyena");
    }
  }
}

torch::Tensor WaveTransformerImpl::forward_backbone(torch::Tensor input_ids) {
  auto x = embeddings_(input_ids); // [B, S, D]
  if (embedding_norm_) x = (*embedding_norm_)(x);

  const MixerType mix = config_.sequence_mixer;

  for (int64_t i = 0; i < config_.n_layers; ++i) {
    bool standard_attn_layer = false;
    if (mix == MixerType::HybridWave || mix == MixerType::HybridHyena) {
      standard_attn_layer = (i % 2 == 1); // odd layers are standard
    }

    if (standard_attn_layer) {
      auto blk = blocks_->ptr<ReorderedNormTransformerBlockImpl>(i);
      x = blk->forward(x);
    } else {
      auto blk = blocks_->ptr<WaveBlockImpl>(i);
      x = blk->forward(x);
    }
  }
  return x; // [B, S, D]
}

torch::Tensor WaveTransformerImpl::forward(
    torch::Tensor input_ids,
    c10::optional<torch::Tensor> labels,
    int64_t ignore_index)
{
  auto hidden = forward_backbone(input_ids);
  auto logits = lm_head_(hidden); // [B, S, vocab_size]

  if (!labels.has_value()) return logits;

  // Cross-entropy loss (same convention as Transformer::forward)
  auto log_probs = torch::log_softmax(logits.view({-1, config_.vocab_size}),
                                       /*dim=*/-1);
  return torch::nll_loss(log_probs,
                          labels->view({-1}),
                          /*weight=*/{},
                          /*reduction=*/at::Reduction::Mean,
                          ignore_index);
}

void WaveTransformerImpl::init_weights(torch::optional<torch::Generator> gen) {
  // Embedding: normal init, std = init_std (consistent with Transformer::init_weights)
  torch::nn::init::normal_(embeddings_->weight, /*mean=*/0.0, config_.init_std);
  // Delegate per-block init to the LM head; individual blocks use default init.
  // (Wave/Hyena params are all small-init or zero-init already in constructors.)
  (void)gen;
}

}  // namespace olmo_cpp
