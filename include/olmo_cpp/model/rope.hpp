#pragma once

#include <torch/torch.h>
#include <optional>
#include <unordered_map>

namespace olmo_cpp {

struct RoPEBuffers {
  torch::Tensor pos_sin;  // (seq_len, head_dim)
  torch::Tensor pos_cos;
};

/// Rotary Position Embedding (RoPE)
/// inv_freq[i] = 1 / (theta^(2i/d))
/// Apply: output = x * cos + rotate_half(x) * sin
class RotaryEmbeddingImpl : public torch::nn::Module {
 public:
  RotaryEmbeddingImpl(int64_t head_size, int64_t theta = 500000);

  RoPEBuffers get_buffers(int64_t seq_len, torch::Device device,
                          torch::Dtype dtype = torch::kFloat32);

  std::pair<torch::Tensor, torch::Tensor> apply(
      torch::Tensor q,
      torch::Tensor k,
      const RoPEBuffers& bufs,
      std::optional<int64_t> start_pos = std::nullopt);

 private:
  int64_t dim_;
  int64_t theta_;
  torch::Tensor compute_inv_freqs(torch::Device device);
  torch::Tensor rotate_half(torch::Tensor x);
  torch::Tensor apply_rotary(torch::Tensor t, torch::Tensor sin, torch::Tensor cos);
};

TORCH_MODULE(RotaryEmbedding);

}  // namespace olmo_cpp
