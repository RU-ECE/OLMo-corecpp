#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/train.hpp"
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <optional>

namespace {
torch::Device select_device(const std::string& preferred) {
  if (preferred == "mps" || preferred == "metal") {
#ifdef __APPLE__
    if (torch::mps::is_available()) {
      std::cout << "Using Metal (MPS) on Apple Silicon\n";
      return torch::Device(torch::kMPS);
    }
#endif
    std::cerr << "MPS requested but not available. Falling back to CPU.\n";
  }
  if (preferred == "cuda") {
    if (torch::cuda::is_available()) {
      std::cout << "Using CUDA GPU\n";
      return torch::Device(torch::kCUDA);
    }
    std::cerr << "CUDA requested but not available. Falling back to CPU.\n";
  }
  std::cout << "Using CPU\n";
  return torch::Device(torch::kCPU);
}
}  // namespace

int main(int argc, char** argv) {
  bool do_train = false;
  std::optional<std::string> data_path;
  std::optional<std::string> config_path;
  std::string device_pref = "auto";
  int64_t batch_size = 4;
  int64_t seq_len = 128;
  int64_t steps = 20;
  double lr = 1e-4;
  int64_t warmup_steps = 100;
  int64_t grad_accum_steps = 1;
  bool use_amp = false;
  std::optional<std::string> checkpoint_path;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--train") {
      do_train = true;
    } else if (arg == "--data-path" && i + 1 < argc) {
      data_path = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device_pref = argv[++i];
    } else if (arg == "--batch-size" && i + 1 < argc) {
      batch_size = std::stoll(argv[++i]);
    } else if (arg == "--seq-len" && i + 1 < argc) {
      seq_len = std::stoll(argv[++i]);
    } else if (arg == "--steps" && i + 1 < argc) {
      steps = std::stoll(argv[++i]);
    } else if (arg == "--lr" && i + 1 < argc) {
      lr = std::stod(argv[++i]);
    } else if (arg == "--warmup-steps" && i + 1 < argc) {
      warmup_steps = std::stoll(argv[++i]);
    } else if (arg == "--grad-accum" && i + 1 < argc) {
      grad_accum_steps = std::stoll(argv[++i]);
    } else if (arg == "--amp") {
      use_amp = true;
    } else if (arg == "--save" && i + 1 < argc) {
      checkpoint_path = argv[++i];
    }
  }

  if (device_pref == "auto") {
#ifdef __APPLE__
    device_pref = torch::mps::is_available() ? "mps" : "cpu";
#else
    device_pref = torch::cuda::is_available() ? "cuda" : "cpu";
#endif
  }

  try {
    olmo_cpp::TransformerConfig cfg;
#ifdef HAS_NLOHMANN_JSON
    if (config_path) {
      cfg = olmo_cpp::load_config_from_json(*config_path);
    } else
#endif
    {
      cfg.d_model = 256;
      cfg.vocab_size = 50257;
      cfg.n_layers = 4;
      cfg.n_heads = 8;
      cfg.rope_theta = 500000;
      cfg.layer_norm_eps = 1e-6;
      cfg.init_std = 0.02;
      cfg.use_qk_norm = true;
    }
    cfg.validate();
    auto model = olmo_cpp::Transformer(cfg);
    model->init_weights();

    auto device = select_device(device_pref);
    model->to(device);

    if (do_train) {
      olmo_cpp::train_epoch(model, cfg, steps, data_path, batch_size, seq_len,
                            lr, warmup_steps, device, grad_accum_steps, use_amp);
      if (checkpoint_path) {
        std::filesystem::create_directories(std::filesystem::path(*checkpoint_path).parent_path());
        torch::save(model, *checkpoint_path);
        std::cout << "Saved checkpoint to " << *checkpoint_path << std::flush << std::endl;
      }
    } else {
      auto input = torch::randint(0, 50257, {2, 128}, torch::TensorOptions().dtype(torch::kLong).device(device));
      auto output = model->forward(input, c10::nullopt);
      std::cout << "OLMo C++ forward pass OK. Output shape: ";
      for (int64_t i = 0; i < output.dim(); ++i) {
        std::cout << output.size(i);
        if (i < output.dim() - 1) std::cout << " x ";
      }
      std::cout << std::endl;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
