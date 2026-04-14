#include "zwt/layers/ffn.hpp"
#include "zwt/ops/elementwise.hpp"

namespace zwt {

FFN::FFN(int64_t d_model, int64_t hidden, DType dtype, Device device,
         uint64_t init_seed)
    : gate_(d_model, hidden, /*bias=*/false, dtype, device, init_seed ^ 0xF00'0001ULL),
      up_  (d_model, hidden, /*bias=*/false, dtype, device, init_seed ^ 0xF00'0002ULL),
      down_(hidden,  d_model, /*bias=*/false, dtype, device, init_seed ^ 0xF00'0003ULL) {}

Tensor FFN::forward(const Tensor& x) {
  saved_gate_ = gate_.forward(x);
  saved_up_   = up_.forward(x);
  saved_silu_mul_ = empty_scratch(saved_gate_.shape(), saved_gate_.dtype(), saved_gate_.device());
  ops::silu_mul(saved_silu_mul_, saved_gate_, saved_up_);
  return down_.forward(saved_silu_mul_);
}

Tensor FFN::backward(const Tensor& grad_y) {
  Tensor grad_h = down_.backward(grad_y);                    // [..., hidden]
  Tensor grad_gate = empty_scratch(saved_gate_.shape(), saved_gate_.dtype(), saved_gate_.device());
  Tensor grad_up   = empty_scratch(saved_up_.shape(),   saved_up_.dtype(),   saved_up_.device());
  ops::silu_mul_backward(grad_h, saved_gate_, saved_up_, grad_gate, grad_up);

  Tensor grad_x_g = gate_.backward(grad_gate);
  Tensor grad_x_u = up_.backward(grad_up);

  // grad_x is the sum of the two projections' grads.
  Tensor grad_x = empty_scratch(grad_x_g.shape(), grad_x_g.dtype(), grad_x_g.device());
  ops::add(grad_x, grad_x_g, grad_x_u);
  return grad_x;
}

void FFN::collect_params(std::vector<Parameter*>& out) {
  gate_.collect_params(out);
  up_.collect_params(out);
  down_.collect_params(out);
}

}  // namespace zwt
