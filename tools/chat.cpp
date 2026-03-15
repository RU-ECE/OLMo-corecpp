/**
 * Interactive CLI chat with a trained OLMo model.
 *
 * Usage:
 *   ./build/chat --checkpoint checkpoints/model.pt --config configs/olmo2_125M.json \
 *     --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <optional>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace {

torch::Device select_device(const std::string& preferred) {
  if (preferred == "mps" || preferred == "metal") {
#ifdef __APPLE__
    if (torch::mps::is_available())
      return torch::Device(torch::kMPS);
#endif
  }
  if (preferred == "cuda") {
    if (torch::cuda::is_available())
      return torch::Device(torch::kCUDA);
  }
  return torch::Device(torch::kCPU);
}

torch::Tensor apply_repetition_penalty(torch::Tensor logits,
                                       const std::vector<int64_t>& token_ids,
                                       double penalty) {
  if (penalty == 1.0 || token_ids.empty()) return logits;
  logits = logits.contiguous().clone();
  auto logits_accessor = logits.accessor<float, 1>();
  for (int64_t id : token_ids) {
    if (id < 0 || id >= logits.size(0)) continue;
    float score = logits_accessor[id];
    logits_accessor[id] = (score > 0) ? score / static_cast<float>(penalty)
                                     : score * static_cast<float>(penalty);
  }
  return logits;
}

int64_t sample_logits(torch::Tensor logits, double temperature,
                      int64_t top_k, double top_p, std::mt19937& gen) {
  if (temperature <= 0) {
    return logits.argmax(-1).item<int64_t>();
  }

  logits = logits / temperature;
  auto logits_cpu = logits.cpu().contiguous();
  int64_t vocab_size = logits_cpu.size(0);

  if (top_k > 0 && top_k < vocab_size) {
    auto [topk_vals, topk_indices] = logits_cpu.topk(top_k);
    auto threshold = topk_vals.index({topk_vals.size(0) - 1}).item<float>();
    logits_cpu = torch::where(logits_cpu < threshold,
                              torch::full_like(logits_cpu, -std::numeric_limits<float>::infinity()),
                              logits_cpu);
  }

  auto probs = torch::softmax(logits_cpu, -1);

  if (top_p < 1.0) {
    auto [sorted_probs, sorted_indices] = probs.sort(-1, true);
    auto cumulative = sorted_probs.cumsum(-1);
    auto mask = cumulative - sorted_probs > top_p;
    sorted_probs.index_put_({mask}, 0.0f);
    probs.zero_();
    probs.scatter_(-1, sorted_indices, sorted_probs);
    auto sum = probs.sum();
    if (sum.item<float>() > 0) {
      probs = probs / sum;
    }
  }

  std::vector<double> probs_vec(vocab_size);
  auto p_ptr = probs.data_ptr<float>();
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs_vec[i] = static_cast<double>(p_ptr[i]);
  }
  std::discrete_distribution<int64_t> dist(probs_vec.begin(), probs_vec.end());
  return dist(gen);
}

}  // namespace

int main(int argc, char** argv) {
  std::string checkpoint_path, config_path, vocab_path, merges_path;
  std::string device_pref = "auto";
  int64_t max_tokens = 128;
  double temperature = 0.8;
  int64_t top_k = 50;
  double top_p = 0.9;
  double repetition_penalty = 1.1;
  bool legacy_decode = false;
  bool use_kv_cache = true;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
    else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
    else if (arg == "--vocab-file" && i + 1 < argc) vocab_path = argv[++i];
    else if (arg == "--merges-file" && i + 1 < argc) merges_path = argv[++i];
    else if (arg == "--device" && i + 1 < argc) device_pref = argv[++i];
    else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::stoll(argv[++i]);
    else if (arg == "--temperature" && i + 1 < argc) temperature = std::stod(argv[++i]);
    else if (arg == "--top-k" && i + 1 < argc) top_k = std::stoll(argv[++i]);
    else if (arg == "--top-p" && i + 1 < argc) top_p = std::stod(argv[++i]);
    else if (arg == "--repetition-penalty" && i + 1 < argc) repetition_penalty = std::stod(argv[++i]);
    else if (arg == "--legacy-decode") legacy_decode = true;
    else if (arg == "--no-kv-cache") use_kv_cache = false;
  }

  if (checkpoint_path.empty() || config_path.empty() || vocab_path.empty() || merges_path.empty()) {
    std::cerr << "Usage: chat --checkpoint <path> --config <path> "
                 "--vocab-file <path> --merges-file <path>\n"
              << "\nOptions:\n"
              << "  --device <mps|cuda|cpu>     (default: auto)\n"
              << "  --max-tokens <n>            (default: 128)\n"
              << "  --temperature <float>       (default: 0.8)\n"
              << "  --top-k <n>                 (default: 50, 0=disabled)\n"
              << "  --top-p <float>             (default: 0.9, 1.0=disabled)\n"
              << "  --repetition-penalty <float>(default: 1.1, 1.0=disabled)\n"
              << "  --legacy-decode            (for checkpoints trained with old tokenizer)\n"
              << "  --no-kv-cache              (slower, avoids MPS memory issues)\n";
    return 1;
  }

  if (device_pref == "auto") {
#ifdef __APPLE__
    device_pref = torch::mps::is_available() ? "mps" : "cpu";
#else
    device_pref = torch::cuda::is_available() ? "cuda" : "cpu";
#endif
  }

  auto device = select_device(device_pref);
  bool is_mps = device.is_mps();

  if (is_mps && use_kv_cache) {
    std::cout << "Note: KV cache disabled on MPS for stability. Using full-context mode.\n";
    use_kv_cache = false;
  }

  try {
#ifdef HAS_NLOHMANN_JSON
    auto cfg = olmo_cpp::load_config_from_json(config_path);
    cfg.validate();

    olmo_cpp::Transformer model(cfg);
    torch::load(model, checkpoint_path);
    model->to(device);
    model->eval();

    olmo_cpp::BPETokenizer tokenizer;
    if (!tokenizer.load(vocab_path, merges_path)) {
      std::cerr << "Failed to load tokenizer\n";
      return 1;
    }
    tokenizer.set_legacy_decode(legacy_decode);

    std::mt19937 rng(std::random_device{}());
    std::cout << "OLMo Chat (type 'quit' to exit)\n" << std::endl;

    while (true) {
      std::cout << "You: ";
      std::string prompt;
      if (!std::getline(std::cin, prompt)) break;
      if (prompt == "quit" || prompt == "exit" || prompt == "q") break;
      if (prompt.empty()) continue;

      std::vector<uint32_t> prompt_ids;
      tokenizer.encode_append(prompt, prompt_ids);

      if (prompt_ids.empty()) {
        std::cout << "Model: (empty)\n" << std::endl;
        continue;
      }

      std::vector<int64_t> all_tokens(prompt_ids.begin(), prompt_ids.end());
      int64_t prompt_len = static_cast<int64_t>(all_tokens.size());
      int64_t max_total = prompt_len + max_tokens;
      if (max_total > 2048) max_total = 2048;

      std::cout << "Model: " << std::flush;

      auto gen_start = std::chrono::steady_clock::now();
      int64_t tokens_generated = 0;

      if (use_kv_cache) {
        olmo_cpp::KVCache kv_cache(model->n_layers());
        {
          auto input = torch::from_blob(all_tokens.data(), {1, prompt_len},
                                        torch::TensorOptions().dtype(torch::kInt64))
                           .clone()
                           .to(device);
          torch::NoGradGuard no_grad;
          auto logits = model->forward(input, c10::nullopt, -100, &kv_cache);
          auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0);
          next_logits = apply_repetition_penalty(next_logits, all_tokens, repetition_penalty);
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p, rng);
          all_tokens.push_back(next_id);
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << tokenizer.decode(tok_to_decode) << std::flush;
          tokens_generated++;
        }
        for (int64_t step = prompt_len + 1; step < max_total; ++step) {
          int64_t last_token = all_tokens.back();
          if (last_token == static_cast<int64_t>(tokenizer.eos_id())) break;
          auto input = torch::tensor({last_token}, torch::kInt64).unsqueeze(0).to(device);
          torch::NoGradGuard no_grad;
          auto logits = model->forward(input, c10::nullopt, -100, &kv_cache);
          auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0);
          next_logits = apply_repetition_penalty(next_logits, all_tokens, repetition_penalty);
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p, rng);
          all_tokens.push_back(next_id);
          if (next_id == static_cast<int64_t>(tokenizer.eos_id())) break;
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << tokenizer.decode(tok_to_decode) << std::flush;
          tokens_generated++;
        }
      } else {
        for (int64_t step = prompt_len; step < max_total; ++step) {
          int64_t last_token = all_tokens.back();
          if (last_token == static_cast<int64_t>(tokenizer.eos_id())) break;
          auto cur_seq_len = static_cast<int64_t>(all_tokens.size());
          auto input_cpu = torch::tensor(
              at::IntArrayRef(all_tokens.data(), static_cast<size_t>(cur_seq_len)),
              torch::kInt64).unsqueeze(0);
          auto input = input_cpu.to(device);
          torch::NoGradGuard no_grad;
          auto logits = model->forward(input, c10::nullopt, -100, nullptr);
#ifdef __APPLE__
          if (is_mps) torch::mps::synchronize();
#endif
          auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0).cpu().contiguous();
          logits.reset();
          input.reset();
          next_logits = apply_repetition_penalty(next_logits, all_tokens, repetition_penalty);
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p, rng);
          all_tokens.push_back(next_id);
          if (next_id == static_cast<int64_t>(tokenizer.eos_id())) break;
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << tokenizer.decode(tok_to_decode) << std::flush;
          tokens_generated++;
        }
      }

      auto gen_end = std::chrono::steady_clock::now();
      double gen_s = std::chrono::duration<double>(gen_end - gen_start).count();
      double tok_per_s = tokens_generated / (gen_s > 0 ? gen_s : 1);

      std::cout << "\n[" << tokens_generated << " tokens, "
                << std::fixed << std::setprecision(1) << tok_per_s << " tok/s]\n"
                << std::endl;
    }
#else
    std::cerr << "Chat requires nlohmann/json\n";
    return 1;
#endif
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
