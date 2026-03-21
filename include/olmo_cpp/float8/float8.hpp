#pragma once

#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

/// Float8 format types
enum class Float8Format {
  E4M3,  // 4 exponent, 3 mantissa (forward pass)
  E5M2   // 5 exponent, 2 mantissa (backward pass)
};

/// Dynamic scale state for float8 quantization
struct Float8ScaleState {
  torch::Tensor amax_history;   // circular buffer of max abs values
  int64_t history_len;
  int64_t current_idx = 0;

  explicit Float8ScaleState(int64_t history_len = 16);

  /// Update amax history with current tensor's absolute max
  void update(const torch::Tensor& tensor);

  /// Get current scale factor based on amax history
  torch::Tensor get_scale(Float8Format format) const;
};

/// Quantized float8 tensor (stored as uint8 + scale)
struct Float8Tensor {
  torch::Tensor data;    // uint8 storage
  torch::Tensor scale;   // fp32 scale factor
  Float8Format format;
  std::vector<int64_t> original_shape;
  torch::ScalarType original_dtype;

  /// Dequantize to bfloat16 or float16
  torch::Tensor dequantize(torch::ScalarType dtype = torch::kBFloat16) const;
};

/// Quantize float tensor to Float8
Float8Tensor quantize_to_float8(const torch::Tensor& tensor, Float8Format format,
                                 Float8ScaleState* state = nullptr);

/// Float8 linear layer: quantizes weights and activations for memory savings
class Float8LinearImpl : public torch::nn::Module {
 public:
  Float8LinearImpl(int64_t in_features, int64_t out_features, bool bias = false);

  torch::Tensor forward(torch::Tensor input);

  void set_float8_enabled(bool enabled) { enabled_ = enabled; }
  bool float8_enabled() const { return enabled_; }

 private:
  torch::nn::Linear inner_;
  bool enabled_ = true;
  Float8ScaleState input_scale_;
  Float8ScaleState weight_scale_;
};
TORCH_MODULE(Float8Linear);

/// MXFP8: microscaling float8 with per-block scales
struct MXFP8Config {
  int64_t block_size = 32;
};

Float8Tensor quantize_mxfp8(const torch::Tensor& tensor, const MXFP8Config& config);

}  // namespace olmo_cpp
