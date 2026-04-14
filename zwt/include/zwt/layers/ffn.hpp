#pragma once

#include "zwt/layers/linear.hpp"

namespace zwt {

// SwiGLU FFN: y = down(silu(gate(x)) * up(x))
class FFN final : public Module {
 public:
  FFN(int64_t d_model, int64_t hidden, DType dtype, Device device,
      uint64_t init_seed = 0xFF77000ULL);

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_y) override;
  void   collect_params(std::vector<Parameter*>& out) override;

 private:
  Linear gate_;
  Linear up_;
  Linear down_;

  Tensor saved_gate_;
  Tensor saved_up_;
  Tensor saved_silu_mul_;  // silu(gate) * up — intermediate activation
};

}  // namespace zwt
