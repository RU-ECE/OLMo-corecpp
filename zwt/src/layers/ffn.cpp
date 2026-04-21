#include "zwt/layers/ffn.hpp"
#include "zwt/ops/elementwise.hpp"

namespace zwt {

FFN::FFN(int64_t d_model, int64_t hidden, DType dtype, Device device,
         uint64_t init_seed)
    // Single fused gate+up projection of out_features = 2*hidden. One GEMM
    // on forward, one GEMM on backward (plus the weight-grad GEMM) instead
    // of two of each. The 2*hidden out_features shape is slightly better
    // for tensor-core tiling too — wider M for cuBLAS.
    : gate_up_(d_model, 2 * hidden, /*bias=*/false, dtype, device,
               init_seed ^ 0xF00'0012ULL),
      down_   (hidden,  d_model,    /*bias=*/false, dtype, device,
               init_seed ^ 0xF00'0003ULL) {}

Tensor FFN::forward(const Tensor& x) {
  saved_combined_ = gate_up_.forward(x);                 // [..., 2*hidden]
  // Output of silu_mul_gated is [..., hidden] — reshape from combined.
  Shape out_shape = saved_combined_.shape();
  out_shape.dims[out_shape.rank - 1] /= 2;
  saved_silu_mul_ = empty_scratch(out_shape, saved_combined_.dtype(),
                                  saved_combined_.device());
  ops::silu_mul_gated(saved_silu_mul_, saved_combined_);
  return down_.forward(saved_silu_mul_);
}

Tensor FFN::backward(const Tensor& grad_y) {
  Tensor grad_h = down_.backward(grad_y);                // [..., hidden]
  Tensor grad_combined = empty_scratch(saved_combined_.shape(),
                                       saved_combined_.dtype(),
                                       saved_combined_.device());
  ops::silu_mul_gated_backward(grad_h, saved_combined_, grad_combined);
  return gate_up_.backward(grad_combined);               // [..., d_model]
}

void FFN::collect_params(std::vector<Parameter*>& out) {
  gate_up_.collect_params(out);
  down_.collect_params(out);
}

}  // namespace zwt
