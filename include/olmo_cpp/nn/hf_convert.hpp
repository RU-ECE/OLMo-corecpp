#pragma once

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace olmo_cpp {

/// Bidirectional mapping between OLMo C++ and HuggingFace param names
struct StateMapping {
  std::unordered_map<std::string, std::string> olmo_to_hf;
  std::unordered_map<std::string, std::string> hf_to_olmo;
  std::vector<std::string> transpose_keys;  // keys needing transpose

  /// Create mapping for OLMo2/Llama-style models
  static StateMapping create_olmo2_mapping(int num_layers, int num_heads, int num_kv_heads);

  /// Create mapping for hybrid models
  static StateMapping create_hybrid_mapping(int num_layers);
};

/// Converts between OLMo C++ and HuggingFace checkpoint formats
class HFConverter {
 public:
  enum class Direction { TO_HF, FROM_HF };

  explicit HFConverter(const StateMapping& mapping);

  /// Convert state dict in-memory
  std::unordered_map<std::string, torch::Tensor> convert(
      const std::unordered_map<std::string, torch::Tensor>& state_dict,
      Direction direction) const;

  /// Load OLMo .pt, save as HF safetensors
  void convert_checkpoint_to_hf(const std::string& olmo_path,
                                 const std::string& hf_output_path,
                                 const std::string& format = "safetensors") const;

  /// Load HF safetensors, save as OLMo .pt
  void convert_checkpoint_from_hf(const std::string& hf_path,
                                   const std::string& olmo_output_path) const;

  /// Write HF config.json
  static void write_hf_config(const std::string& output_path,
                               int vocab_size, int hidden_size, int num_layers,
                               int num_heads, int num_kv_heads, int intermediate_size,
                               double rope_theta, const std::string& model_type = "llama");

  /// Write HF tokenizer_config.json
  static void write_tokenizer_config(const std::string& output_path,
                                      const std::string& tokenizer_class = "GPT2Tokenizer");

 private:
  StateMapping mapping_;
};

/// Safetensors format reader/writer
namespace safetensors {
std::unordered_map<std::string, torch::Tensor> load(const std::string& path);
void save(const std::unordered_map<std::string, torch::Tensor>& state_dict,
          const std::string& path);
}  // namespace safetensors

}  // namespace olmo_cpp
