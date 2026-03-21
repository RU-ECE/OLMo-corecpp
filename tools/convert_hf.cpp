/**
 * Convert HuggingFace OLMo-2 safetensors → OLMo C++ .pt checkpoint
 *
 * Usage:
 *   ./build/convert_hf --hf-dir <path_to_hf_model> \
 *       --config configs/olmo2_7B_mtp.json \
 *       --output checkpoints/olmo2_7B_mtp.pt
 *
 * This loads HF safetensors, maps parameter names to our C++ module names,
 * creates the model (with randomly initialized MTP heads if configured),
 * loads backbone weights, and saves the full model.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/nn/hf_convert.hpp"
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace fs = std::filesystem;

namespace {

/// Map HF OLMo-2 / Llama-style param names to our module hierarchy.
/// Our model names: embeddings.weight, embedding_norm.weight,
///   blocks.{i}.attention.{w_q,w_k,w_v,w_out}.weight
///   blocks.{i}.attention.{q_norm,k_norm}.weight
///   blocks.{i}.attention_norm.weight, blocks.{i}.feed_forward_norm.weight
///   blocks.{i}.feed_forward.{w1,w2,w3}.weight
///   lm_head.norm.weight, lm_head.w_out.weight
///   mtp_heads.{k}.proj.weight, mtp_heads.{k}.norm.weight
std::string hf_to_olmo_name(const std::string& hf_name) {
  // Embeddings
  if (hf_name == "model.embed_tokens.weight") return "embeddings.weight";

  // Final norm (HF) → embedding_norm (ours — OLMo2 uses post-embedding norm)
  if (hf_name == "model.norm.weight") return "lm_head.norm.weight";

  // LM head
  if (hf_name == "lm_head.weight") return "lm_head.w_out.weight";

  // Layer params: model.layers.{i}.xxx → blocks.{i}.xxx
  if (hf_name.substr(0, 13) == "model.layers.") {
    // Extract layer index
    auto rest = hf_name.substr(13);
    auto dot = rest.find('.');
    if (dot == std::string::npos) return hf_name;
    std::string layer_idx = rest.substr(0, dot);
    std::string suffix = rest.substr(dot + 1);
    std::string prefix = "blocks." + layer_idx;

    // Attention projections
    if (suffix == "self_attn.q_proj.weight") return prefix + ".attention.w_q.weight";
    if (suffix == "self_attn.k_proj.weight") return prefix + ".attention.w_k.weight";
    if (suffix == "self_attn.v_proj.weight") return prefix + ".attention.w_v.weight";
    if (suffix == "self_attn.o_proj.weight") return prefix + ".attention.w_out.weight";

    // QK norms
    if (suffix == "self_attn.q_norm.weight") return prefix + ".attention.q_norm.weight";
    if (suffix == "self_attn.k_norm.weight") return prefix + ".attention.k_norm.weight";

    // FFN
    if (suffix == "mlp.gate_proj.weight") return prefix + ".feed_forward.w1.weight";
    if (suffix == "mlp.up_proj.weight") return prefix + ".feed_forward.w3.weight";
    if (suffix == "mlp.down_proj.weight") return prefix + ".feed_forward.w2.weight";

    // Layer norms
    if (suffix == "input_layernorm.weight") return prefix + ".attention_norm.weight";
    if (suffix == "post_attention_layernorm.weight") return prefix + ".feed_forward_norm.weight";

    return hf_name;  // unmapped
  }

  return hf_name;  // unmapped
}

}  // namespace

int main(int argc, char** argv) {
  std::string hf_dir, config_path, output_path;
  bool fp32 = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--hf-dir" && i + 1 < argc) hf_dir = argv[++i];
    else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
    else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    else if (arg == "--fp32") fp32 = true;
  }

  if (hf_dir.empty() || config_path.empty() || output_path.empty()) {
    std::cerr << "Usage: convert_hf --hf-dir <hf_model_dir> "
                 "--config <config.json> --output <output.pt> [--fp32]\n";
    return 1;
  }

#ifdef HAS_NLOHMANN_JSON
  try {
    // Load our config
    auto cfg = olmo_cpp::load_config_from_json(config_path);
    cfg.validate();

    std::cout << "Model config: d_model=" << cfg.d_model
              << " n_layers=" << cfg.n_layers
              << " n_heads=" << cfg.n_heads
              << " n_kv_heads=" << cfg.get_n_kv_heads()
              << " num_mtp_heads=" << cfg.num_mtp_heads << "\n";

    // Create model (MTP heads will be randomly initialized)
    auto model = olmo_cpp::Transformer(cfg);
    model->init_weights();

    // Find all safetensors files in the HF directory
    std::vector<std::string> st_files;
    for (const auto& entry : fs::directory_iterator(hf_dir)) {
      if (entry.path().extension() == ".safetensors") {
        st_files.push_back(entry.path().string());
      }
    }
    std::sort(st_files.begin(), st_files.end());

    if (st_files.empty()) {
      std::cerr << "No .safetensors files found in " << hf_dir << "\n";
      return 1;
    }
    std::cout << "Found " << st_files.size() << " safetensors file(s)\n";

    // Build a map of our model's parameter names → tensor references
    std::unordered_map<std::string, torch::Tensor*> our_params;
    for (auto& item : model->named_parameters()) {
      our_params[item.key()] = &item.value();
    }

    int loaded = 0, skipped = 0;

    // Load each safetensors file
    for (const auto& st_file : st_files) {
      std::cout << "Loading " << fs::path(st_file).filename().string() << "...\n";
      auto tensors = olmo_cpp::safetensors::load(st_file);

      for (const auto& [hf_name, hf_tensor] : tensors) {
        std::string our_name = hf_to_olmo_name(hf_name);
        auto it = our_params.find(our_name);
        if (it != our_params.end()) {
          auto target = fp32 ? hf_tensor.to(torch::kFloat32) : hf_tensor;
          // Check shape match
          if (it->second->sizes() != target.sizes()) {
            std::cerr << "  Shape mismatch: " << our_name
                      << " expected " << it->second->sizes()
                      << " got " << target.sizes() << " — skipping\n";
            skipped++;
            continue;
          }
          torch::NoGradGuard no_grad;
          it->second->copy_(target);
          loaded++;
        } else {
          std::cout << "  Unmapped: " << hf_name << " → " << our_name << "\n";
          skipped++;
        }
      }
    }

    std::cout << "\nLoaded " << loaded << " params, skipped " << skipped << "\n";

    if (cfg.num_mtp_heads > 0) {
      std::cout << "MTP heads (" << cfg.num_mtp_heads
                << ") initialized randomly — train them on your data\n";
    }

    // Save
    fs::create_directories(fs::path(output_path).parent_path());
    torch::save(model, output_path);
    std::cout << "Saved to " << output_path << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
#else
  std::cerr << "convert_hf requires nlohmann/json\n";
  return 1;
#endif
}
