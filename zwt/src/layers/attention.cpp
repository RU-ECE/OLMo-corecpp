#include "zwt/layers/attention.hpp"
#include "zwt/core/allocator.hpp"
#include "zwt/ops/attn.hpp"
#include "zwt/ops/elementwise.hpp"
#include "zwt/ops/rope.hpp"

#include <stdexcept>

namespace zwt {

namespace {

// Project q/k/v independently — three separate GEMMs. Q/K/V share the seed
// space but use disjoint sub-seeds so their weight tensors differ.
constexpr uint64_t kSeedSaltQ = 0x17'00'00'00ULL;
constexpr uint64_t kSeedSaltK = 0x23'00'00'00ULL;
constexpr uint64_t kSeedSaltV = 0x31'00'00'00ULL;
constexpr uint64_t kSeedSaltO = 0x47'00'00'00ULL;

}  // namespace

Attention::Attention(const Config& cfg, DType dtype, Device device,
                     uint64_t init_seed)
    : cfg_(cfg),
      q_proj_(cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.bias, dtype, device,
              init_seed ^ kSeedSaltQ),
      k_proj_(cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.bias, dtype, device,
              init_seed ^ kSeedSaltK),
      v_proj_(cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.bias, dtype, device,
              init_seed ^ kSeedSaltV),
      out_proj_(cfg.n_heads * cfg.head_dim, cfg.d_model, cfg.bias, dtype, device,
                init_seed ^ kSeedSaltO) {
  if (cfg.d_model <= 0 || cfg.n_heads <= 0 || cfg.head_dim <= 0 || cfg.max_seq <= 0)
    throw std::runtime_error("Attention: invalid config");
  if (cfg.d_model != cfg.n_heads * cfg.head_dim)
    throw std::runtime_error("Attention: d_model must equal n_heads * head_dim");
  rope_table_ = ops::rope_build_table(cfg.max_seq, cfg.head_dim, cfg.rope_base, device);
}

Tensor Attention::forward(const Tensor& x) {
  if (x.rank() != 3) throw std::runtime_error("Attention::forward: x must be [B,S,d]");
  const int64_t B = x.dim(0);
  const int64_t S = x.dim(1);
  const int64_t H = cfg_.n_heads;
  const int64_t D = cfg_.head_dim;
  saved_input_ = x.view(x.shape());

  // Three separate projections.
  Tensor q = q_proj_.forward(x);  // [B, S, H*D]
  Tensor k = k_proj_.forward(x);
  Tensor v = v_proj_.forward(x);

  Tensor q4 = q.view({B, S, H, D});
  Tensor k4 = k.view({B, S, H, D});
  Tensor v4 = v.view({B, S, H, D});

  // RoPE on Q and K (not V).
  ops::rope_apply(q4, rope_table_);
  ops::rope_apply(k4, rope_table_);

  // Transpose to head-major for SDPA: [B,S,H,D] -> [B,H,S,D].
  saved_q_bhsd_ = empty_scratch({B, H, S, D}, x.dtype(), x.device());
  saved_k_bhsd_ = empty_scratch({B, H, S, D}, x.dtype(), x.device());
  saved_v_bhsd_ = empty_scratch({B, H, S, D}, x.dtype(), x.device());
  ops::transpose_bshd_to_bhsd(q4, saved_q_bhsd_);
  ops::transpose_bshd_to_bhsd(k4, saved_k_bhsd_);
  ops::transpose_bshd_to_bhsd(v4, saved_v_bhsd_);

  saved_out_bhsd_ = empty_scratch({B, H, S, D}, x.dtype(), x.device());
  ops::sdpa(saved_q_bhsd_, saved_k_bhsd_, saved_v_bhsd_,
            saved_out_bhsd_, /*is_causal=*/true);

  // Transpose back to [B, S, H, D], flatten last two dims for out_proj.
  Tensor out_bshd = empty_scratch({B, S, H, D}, x.dtype(), x.device());
  ops::transpose_bhsd_to_bshd(saved_out_bhsd_, out_bshd);
  Tensor out_flat = out_bshd.view({B, S, H * D});

  return out_proj_.forward(out_flat);
}

Tensor Attention::backward(const Tensor& grad_y) {
  const int64_t B = saved_input_.dim(0);
  const int64_t S = saved_input_.dim(1);
  const int64_t H = cfg_.n_heads;
  const int64_t D = cfg_.head_dim;

  // Backward through out_proj.
  Tensor grad_out_flat = out_proj_.backward(grad_y);                   // [B,S,H*D]
  Tensor grad_out_bshd = grad_out_flat.view({B, S, H, D});

  // Transpose grad [B,S,H,D] -> [B,H,S,D].
  Tensor grad_out_bhsd = empty_scratch({B, H, S, D}, grad_y.dtype(), grad_y.device());
  ops::transpose_bshd_to_bhsd(grad_out_bshd, grad_out_bhsd);

  // SDPA backward.
  Tensor grad_q_bhsd = empty_scratch({B, H, S, D}, grad_y.dtype(), grad_y.device());
  Tensor grad_k_bhsd = empty_scratch({B, H, S, D}, grad_y.dtype(), grad_y.device());
  Tensor grad_v_bhsd = empty_scratch({B, H, S, D}, grad_y.dtype(), grad_y.device());
  ops::sdpa_backward(grad_out_bhsd,
                     saved_q_bhsd_, saved_k_bhsd_, saved_v_bhsd_,
                     saved_out_bhsd_,
                     grad_q_bhsd, grad_k_bhsd, grad_v_bhsd,
                     /*is_causal=*/true);

  // Transpose each grad from [B,H,S,D] back to [B,S,H,D].
  Tensor grad_q_bshd = empty_scratch({B, S, H, D}, grad_y.dtype(), grad_y.device());
  Tensor grad_k_bshd = empty_scratch({B, S, H, D}, grad_y.dtype(), grad_y.device());
  Tensor grad_v_bshd = empty_scratch({B, S, H, D}, grad_y.dtype(), grad_y.device());
  ops::transpose_bhsd_to_bshd(grad_q_bhsd, grad_q_bshd);
  ops::transpose_bhsd_to_bshd(grad_k_bhsd, grad_k_bshd);
  ops::transpose_bhsd_to_bshd(grad_v_bhsd, grad_v_bshd);

  // RoPE backward on Q and K grads.
  ops::rope_apply_backward(grad_q_bshd, rope_table_);
  ops::rope_apply_backward(grad_k_bshd, rope_table_);

  // Flatten [B,S,H,D] -> [B,S,H*D] and hand to projection backwards. Each
  // projection writes into grad_x (summed) via a scratch temporary.
  Tensor gq_flat = grad_q_bshd.view({B, S, H * D});
  Tensor gk_flat = grad_k_bshd.view({B, S, H * D});
  Tensor gv_flat = grad_v_bshd.view({B, S, H * D});

  Tensor grad_x_q = q_proj_.backward(gq_flat);   // [B, S, d_model]
  Tensor grad_x_k = k_proj_.backward(gk_flat);
  Tensor grad_x_v = v_proj_.backward(gv_flat);

  // Sum the three upstream grads.
  Tensor grad_x = empty_scratch(saved_input_.shape(), grad_y.dtype(), grad_y.device());
  ops::add(grad_x, grad_x_q, grad_x_k);
  {
    // grad_x += grad_x_v  (use axpy with alpha=1).
    ops::axpy(grad_x, grad_x_v, 1.0f);
  }
  return grad_x;
}

void Attention::collect_params(std::vector<Parameter*>& out) {
  q_proj_.collect_params(out);
  k_proj_.collect_params(out);
  v_proj_.collect_params(out);
  out_proj_.collect_params(out);
}

}  // namespace zwt
