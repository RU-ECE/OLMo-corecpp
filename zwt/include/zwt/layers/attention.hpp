#pragma once

#include "zwt/layers/linear.hpp"
#include "zwt/layers/module.hpp"

namespace zwt {

// Multi-head self-attention with RoPE and causal mask.
//
// Forward: x [B, S, d_model] -> y [B, S, d_model].
//   qkv projected through three separate Linears (q, k, v), reshaped to
//   [B, S, H, D], RoPE applied to q/k, transposed to [B, H, S, D], passed
//   to sdpa() which returns [B, H, S, D], transposed back, reshaped to
//   [B, S, d_model] and fed through the out projection.
//
// Notes on GQA: v1 supports MHA only (n_kv_heads == n_heads). The param
// layout and RoPE table are shape-agnostic enough to extend later.
class Attention final : public Module {
 public:
  struct Config {
    int64_t d_model   = 0;
    int64_t n_heads   = 0;
    int64_t head_dim  = 0;   // must divide d_model and equal d_model / n_heads
    int64_t max_seq   = 0;   // required to size the RoPE table
    float   rope_base = 10000.f;
    bool    bias      = false;
  };

  Attention(const Config& cfg, DType dtype, Device device,
            uint64_t init_seed = 0xA77EBADULL);

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_y) override;
  void   collect_params(std::vector<Parameter*>& out) override;

  const Config& config() const { return cfg_; }

 private:
  Config  cfg_;
  Linear  q_proj_;
  Linear  k_proj_;
  Linear  v_proj_;
  Linear  out_proj_;
  Tensor  rope_table_;       // [max_seq, head_dim] fp32

  // Saved activations (for backward — reused from step_begin arena).
  Tensor  saved_input_;      // view of x, [B, S, d_model]
  Tensor  saved_q_bhsd_;     // [B, H, S, D] — post-RoPE, post-transpose
  Tensor  saved_k_bhsd_;     // [B, H, S, D]
  Tensor  saved_v_bhsd_;     // [B, H, S, D]
  Tensor  saved_out_bhsd_;   // [B, H, S, D] — sdpa output
};

}  // namespace zwt
