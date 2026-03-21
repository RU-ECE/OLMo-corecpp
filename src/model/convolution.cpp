#include "olmo_cpp/model/convolution.hpp"

namespace olmo_cpp {

CausalConv1dImpl::CausalConv1dImpl(int64_t channels, int64_t kernel_size)
    : conv_(register_module("conv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(channels, channels, kernel_size)
            .groups(channels)  // depthwise
            .bias(false)))),
      padding_(kernel_size - 1) {}

torch::Tensor CausalConv1dImpl::forward(torch::Tensor x) {
  // x: [B, S, D] -> transpose to [B, D, S] for Conv1d
  auto out = x.transpose(1, 2);

  // Left-pad to make convolution causal
  out = torch::nn::functional::pad(out,
      torch::nn::functional::PadFuncOptions({padding_, 0}));

  out = conv_(out);

  // Slice to original length and transpose back
  out = out.narrow(2, 0, x.size(1));
  return out.transpose(1, 2);
}

}  // namespace olmo_cpp
