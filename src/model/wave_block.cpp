#include "olmo_cpp/model/wave_block.hpp"

namespace olmo_cpp {

WaveBlockImpl::WaveBlockImpl(const TransformerConfig& cfg,
                               int64_t block_idx,
                               bool use_wave_mixer)
    : use_wave_mixer_(use_wave_mixer),
      mixer_norm_(register_module(
          "mixer_norm",
          RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_norm_(register_module(
          "feed_forward_norm",
          RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_(register_module(
          "feed_forward",
          FeedForward(cfg.d_model, cfg.get_hidden_size())))
{
  if (use_wave_mixer_) {
    wave_attn_ = register_module("wave_attn", WaveAttention(cfg, block_idx));
  } else {
    hyena_ssm_ = register_module("hyena_ssm", HyenaSSM(cfg, block_idx));
  }
}

torch::Tensor WaveBlockImpl::forward(torch::Tensor x) {
  // Reordered-norm residual: h = x + mixer_norm(mixer(x))
  torch::Tensor mixed;
  if (use_wave_mixer_) {
    mixed = wave_attn_->forward(x);
  } else {
    mixed = hyena_ssm_->forward(x);
  }
  auto h = x + mixer_norm_(mixed);

  // h = h + ff_norm(ff(h))
  return h + feed_forward_norm_(feed_forward_(h));
}

}  // namespace olmo_cpp
