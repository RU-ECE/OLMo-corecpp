#pragma once

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/block.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/lm_head.hpp"
#include <torch/torch.h>
#include <optional>
#include <vector>

namespace olmo_cpp {

class TransformerImpl : public torch::nn::Module {
 public:
  explicit TransformerImpl(const TransformerConfig& cfg);

  /// Forward pass. When kv_cache is provided, uses incremental decoding.
  /// When labels are provided, returns cross-entropy loss.
  torch::Tensor forward(
      torch::Tensor input_ids,
      c10::optional<torch::Tensor> labels = c10::nullopt,
      int64_t ignore_index = -100,
      KVCache* kv_cache = nullptr);

  void init_weights(torch::optional<torch::Generator> gen = c10::nullopt);

  std::vector<RoPEBuffers> get_rope_buffers(int64_t seq_len, torch::Device device,
                                             torch::Dtype dtype = torch::kFloat32);

  int64_t n_layers() const { return config_.n_layers; }

 private:
  torch::nn::Embedding embeddings_;
  std::optional<RMSNorm> embedding_norm_;
  torch::nn::ModuleList blocks_;
  LMHead lm_head_;
  std::optional<double> embed_scale_;
  TransformerConfig config_;

  // Cached RoPE — recomputed only when seq_len grows or dtype changes
  int64_t cached_rope_len_ = 0;
  torch::Dtype cached_rope_dtype_ = torch::kFloat32;
  std::vector<RoPEBuffers> cached_rope_bufs_;
};

TORCH_MODULE(Transformer);

}  // namespace olmo_cpp
