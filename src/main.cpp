/**
 * OLMo C++ Training
 *
 * Usage: ./build/olmo_train conf/olmo.conf
 */

#include "olmo_cpp/common/config_ini.hpp"
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/fused_transformer.hpp"
#include "olmo_cpp/model/mup_init.hpp"
#include "olmo_cpp/train.hpp"
#include "olmo_cpp/seed.hpp"
#include "olmo_cpp/profiler.hpp"
#include "olmo_cpp/backend/simd_backend.hpp"
#include "olmo_cpp/backend/cuda_backend.hpp"
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

torch::Device select_device(const std::string& preferred) {
  if (preferred == "mps" || preferred == "metal") {
#ifdef __APPLE__
    if (torch::mps::is_available()) {
      std::cout << "Device: Metal (MPS) on Apple Silicon\n";
      return torch::Device(torch::kMPS);
    }
#endif
    std::cerr << "MPS not available, falling back to CPU.\n";
  }
  if (preferred == "cuda") {
    if (torch::cuda::is_available()) {
      std::cout << "Device: CUDA GPU\n";
      return torch::Device(torch::kCUDA);
    }
    std::cerr << "CUDA not available, falling back to CPU.\n";
  }
  std::cout << "Device: CPU\n";
  return torch::Device(torch::kCPU);
}

int64_t count_parameters(const torch::nn::Module& model) {
  int64_t total = 0;
  for (const auto& p : model.parameters()) total += p.numel();
  return total;
}

template<typename Model>
void run(Model& model, const olmo_cpp::TransformerConfig& cfg,
         const olmo_cpp::TrainConfig& train_cfg, torch::Device device,
         bool use_mup, bool use_fused, bool use_multi_res,
         bool enable_profile, const std::string& save_path,
         const olmo_cpp::SeedState& seed_state) {
  // Init weights
  if (use_mup) {
    olmo_cpp::MuPConfig mup_cfg;
    mup_cfg.base_width = 256.0;
    mup_cfg.target_width = static_cast<double>(cfg.d_model);
    olmo_cpp::apply_mup_init(*model, cfg, mup_cfg, seed_state.torch_gen);
  } else {
    model->init_weights(seed_state.torch_gen);
  }

  int64_t n_params = count_parameters(*model);
  std::cout << "Model: " << (n_params / 1000000) << "M params"
            << " (d=" << cfg.d_model
            << ", layers=" << cfg.n_layers
            << ", heads=" << cfg.n_heads;
  if (use_fused) std::cout << ", FUSED";
  if (use_mup) std::cout << ", µP";
  if (use_multi_res) std::cout << ", DC-MRE";
  if (cfg.num_mtp_heads > 0) std::cout << ", mtp_heads=" << cfg.num_mtp_heads;
  std::cout << ")\n";

  model->to(device);
  olmo_cpp::train(model, cfg, train_cfg, device);

  if (enable_profile) {
    olmo_cpp::profiler().report("Training Profile");
    olmo_cpp::print_memory_summary(device);
    olmo_cpp::print_rng_state_summary();
  }

  if (!save_path.empty()) {
    std::filesystem::create_directories(
        std::filesystem::path(save_path).parent_path());
    torch::save(model, save_path);
    std::cout << "Checkpoint saved: " << save_path << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <conf_file>\n"
              << "  e.g.: " << argv[0] << " conf/olmo.conf\n";
    return (argc == 1) ? 0 : 1;
  }

  const std::string conf_path = argv[1];

  try {
    // ── Load all sections from single .conf file ──
    ConfigINI model_ini(conf_path, "model");
    ConfigINI train_ini(conf_path, "training");
    ConfigINI data_ini(conf_path, "data");
    ConfigINI opt_ini(conf_path, "optimization");
    ConfigINI dev_ini(conf_path, "device");

    // ── Model config ──
    olmo_cpp::TransformerConfig cfg;
    model_ini.get("d_model", cfg.d_model);
    model_ini.get("vocab_size", cfg.vocab_size);
    model_ini.get("n_layers", cfg.n_layers);
    model_ini.get("n_heads", cfg.n_heads);
    cfg.n_kv_heads    = model_ini.get_or<int64_t>("n_kv_heads", -1);
    cfg.head_dim      = model_ini.get_or<int64_t>("head_dim", -1);
    cfg.rope_theta    = model_ini.get_or<int64_t>("rope_theta", 500000);
    cfg.layer_norm_eps = model_ini.get_or<double>("layer_norm_eps", 1e-6);
    cfg.init_std      = model_ini.get_or<double>("init_std", 0.02);
    cfg.use_qk_norm   = model_ini.get_or<bool>("use_qk_norm", true);
    cfg.num_mtp_heads = model_ini.get_or<int64_t>("num_mtp_heads", 0);
    cfg.mtp_loss_weight = model_ini.get_or<double>("mtp_loss_weight", 0.1);

    // ── Training config ──
    olmo_cpp::TrainConfig train_cfg;
    train_ini.get("steps", train_cfg.num_steps);
    train_ini.get("batch_size", train_cfg.batch_size);
    train_ini.get("seq_len", train_cfg.seq_len);
    train_ini.get("lr", train_cfg.lr);
    train_cfg.warmup_steps     = train_ini.get_or<int64_t>("warmup_steps", 100);
    train_cfg.grad_accum_steps = train_ini.get_or<int64_t>("grad_accum", 1);
    train_cfg.optimizer        = train_ini.get_or<std::string>("optimizer", "adamw");
    train_cfg.use_amp          = train_ini.get_or<bool>("amp", false);

    int64_t seed_val       = train_ini.get_or<int64_t>("seed", 42);
    bool enable_profile    = train_ini.get_or<bool>("profile", false);
    std::string save_path  = train_ini.get_or<std::string>("save", "");

    // ── Data config ──
    std::string data_path;
    data_ini.get("data_path", data_path);
    train_cfg.data_path = data_path;

    std::string bpe_vocab = data_ini.get_or<std::string>("bpe_vocab", "");

    // ── Optimization flags ──
    bool use_fused     = opt_ini.get_or<bool>("fused", false);
    bool use_mup       = opt_ini.get_or<bool>("mup", false);
    bool use_multi_res = opt_ini.get_or<bool>("multi_res", false);

    // ── Device ──
    std::string device_pref = dev_ini.get_or<std::string>("device", "auto");

    // ── Apply multi-res DC-MRE settings ──
    if (use_multi_res) {
      cfg.use_multi_res = true;
      cfg.bpe_vocab_path = bpe_vocab;
    }

    cfg.validate();

    // Resolve "auto" device
    if (device_pref == "auto") {
#ifdef __APPLE__
      device_pref = torch::mps::is_available() ? "mps" : "cpu";
#else
      device_pref = torch::cuda::is_available() ? "cuda" : "cpu";
#endif
    }

    // Seed all RNGs
    auto seed_state = olmo_cpp::seed_all(static_cast<uint64_t>(seed_val));

    auto device = select_device(device_pref);

    // Select backend
    if (device.is_cuda()) {
      olmo_cpp::use_cuda_backend();
      std::cout << "Backend: CUDA (fused kernels)\n";
    } else if (device.is_cpu()) {
      olmo_cpp::use_simd_backend();
      std::cout << "Backend: SIMD (fused kernels + arena allocator)\n";
    }

    std::cout << "Config: " << conf_path << "\n";

    // Create and train model
    if (use_fused) {
      auto model = olmo_cpp::FusedTransformer(cfg);
      run(model, cfg, train_cfg, device, use_mup, true,
          use_multi_res, enable_profile, save_path, seed_state);
    } else {
      auto model = olmo_cpp::Transformer(cfg);
      run(model, cfg, train_cfg, device, use_mup, false,
          use_multi_res, enable_profile, save_path, seed_state);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
