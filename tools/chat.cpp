/**
 * tools/chat.cpp
 *
 * Interactive CLI for sampling from a trained OLMo C++ model. Loads the
 * Transformer + tokenizer, then in a REPL loop reads "You:" prompts on
 * stdin, encodes them, runs autoregressive decoding (with optional KV
 * cache and MTP speculative decoding), and streams decoded tokens to
 * stdout as they are generated. Per-turn it also prints a stats line
 * with tokens/sec. No files are written.
 *
 * Example:
 *   ./build/chat --checkpoint checkpoints/125M.pt \
 *                --config configs/olmo2_125M.json \
 *                --vocab-file data/gpt2/vocab.json \
 *                --merges-file data/gpt2/merges.txt
 *
 * --- Flags ---
 *   --checkpoint            torch::save'd .pt file produced by training
 *   --config                JSON config used to build the model topology
 *   --vocab-file            GPT-2 vocab.json
 *   --merges-file           GPT-2 merges.txt
 *   --structural-config     optional structural tokenizer config dir
 *   --device                mps | cuda | cpu (default: auto-detect)
 *   --max-tokens            cap on generated tokens per turn (default 128)
 *   --temperature           sampling temperature; <=0 means greedy (0.8)
 *   --top-k                 top-k cutoff, 0 disables (default 50)
 *   --top-p                 nucleus sampling cutoff, 1.0 disables (0.9)
 *   --repetition-penalty    >1.0 penalises tokens already in context (1.1)
 *   --legacy-decode         decode token 33 as space (older tokenizer)
 *   --no-kv-cache           force full-context recompute every step
 *   --no-speculative        disable MTP speculative decoding path
 *
 * --- Build target ---
 *   chat (CMakeLists.txt:517). Links the static `olmo_cpp` library
 *   (model + tokenizer + backends), LibTorch, and nlohmann/json.
 *   Compiled with -O3 -march=native; HAS_NLOHMANN_JSON is defined when
 *   JSON support was found at configure time (required to load configs).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp               : load_config_from_json()
 *   - olmo_cpp/model/transformer.hpp    : Transformer module class
 *   - olmo_cpp/model/kv_cache.hpp       : per-layer KV cache + snapshot/rollback
 *   - olmo_cpp/backend/cuda_graph.hpp   : CUDA graph capture (future use here)
 *   - olmo_cpp/data/bpe_tokenizer.hpp   : GPT-2 BPE
 *   - olmo_cpp/data/structural_tokenizer.hpp : optional structural tokenizer
 *   - olmo_cpp/backend/cuda_backend.hpp : enable fused CUDA kernels on GPU
 *   - olmo_cpp/backend/simd_backend.hpp : enable SIMD CPU kernels on CPU
 *
 * --- Reads / Writes ---
 *   - reads:  checkpoint .pt, config .json, vocab.json, merges.txt,
 *             optional structural-config dir
 *   - writes: nothing — generation is streamed to stdout.
 *
 * --- Role in workflow ---
 *   Used after training (`olmo_train conf/...`) to qualitatively eyeball
 *   model behaviour and benchmark inference throughput. With MTP heads
 *   enabled in the config, this is also where the speculative decoding
 *   acceptance rate is observed.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/backend/cuda_graph.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include "olmo_cpp/data/structural_tokenizer.hpp"
#include "olmo_cpp/backend/cuda_backend.hpp"
#include "olmo_cpp/backend/simd_backend.hpp"
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

/// Resolve the runtime device given a user preference string.
/// Falls through to CPU if MPS/CUDA was requested but isn't available.
/// On Apple Silicon "metal" is treated as an alias for "mps".
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

/// Apply repetition penalty to logits for tokens that already appeared.
/// Returns contiguous tensor (avoids modifying views that can cause segfaults).
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

/// Sample from logits with temperature, top-k, and top-p (nucleus) filtering.
int64_t sample_logits(torch::Tensor logits, double temperature,
                      int64_t top_k, double top_p, std::mt19937& gen) {
  // Greedy
  if (temperature <= 0) {
    return logits.argmax(-1).item<int64_t>();
  }

  // Temperature scaling
  logits = logits / temperature;

  // Move to CPU for sampling
  auto logits_cpu = logits.cpu().contiguous();
  int64_t vocab_size = logits_cpu.size(0);

  // Top-k filtering
  if (top_k > 0 && top_k < vocab_size) {
    auto [topk_vals, topk_indices] = logits_cpu.topk(top_k);
    auto threshold = topk_vals.index({topk_vals.size(0) - 1}).item<float>();
    logits_cpu = torch::where(logits_cpu < threshold,
                              torch::full_like(logits_cpu, -std::numeric_limits<float>::infinity()),
                              logits_cpu);
  }

  // Softmax to get probabilities
  auto probs = torch::softmax(logits_cpu, -1);

  // Top-p (nucleus) filtering
  if (top_p < 1.0) {
    auto [sorted_probs, sorted_indices] = probs.sort(-1, /*descending=*/true);
    auto cumulative = sorted_probs.cumsum(-1);

    // Find cutoff: zero out tokens after cumulative probability exceeds top_p
    auto mask = cumulative - sorted_probs > top_p;
    sorted_probs.index_put_({mask}, 0.0f);

    // Scatter back to original order
    probs.zero_();
    probs.scatter_(-1, sorted_indices, sorted_probs);

    // Renormalize
    auto sum = probs.sum();
    if (sum.item<float>() > 0) {
      probs = probs / sum;
    }
  }

  // Copy to std::vector before discrete_distribution to avoid holding tensor
  // references (fixes MPS malloc/free crashes on Apple Silicon)
  std::vector<double> probs_vec(vocab_size);
  auto p_ptr = probs.data_ptr<float>();
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs_vec[i] = static_cast<double>(p_ptr[i]);
  }
  std::discrete_distribution<int64_t> dist(probs_vec.begin(), probs_vec.end());
  return dist(gen);
}

/// Speculative decoding with KV cache: draft k tokens via MTP heads, verify in batch.
///
/// Algorithm (per step):
///   1. Run backbone incrementally (1 token) → get hidden state + update KV cache
///   2. Main LM head → token t+1
///   3. MTP heads → draft tokens t+2..t+k+1 (no forward pass, just projections)
///   4. Snapshot KV cache, then verify [main_token, drafts...] in one forward pass
///   5. Accept matching tokens, rollback KV cache to first rejection point
///
/// Speedup: O(1) draft (projections only) + O(k) verify vs O(k) full forwards.
/// With k=3 MTP heads: up to 4 tokens from 2 short forward passes instead of 4 full passes.
int64_t speculative_decode_step(
    olmo_cpp::Transformer& model,
    std::vector<int64_t>& all_tokens,
    olmo_cpp::KVCache& kv_cache,
    torch::Device device,
    double temperature,
    int64_t top_k,
    double top_p,
    double repetition_penalty,
    std::mt19937& rng,
    olmo_cpp::BPETokenizer& tokenizer,
    int64_t& total_drafted,
    int64_t& total_accepted) {

  int64_t num_drafts = model->num_mtp_heads();
  int64_t eos_id = static_cast<int64_t>(tokenizer.eos_id());
  torch::NoGradGuard no_grad;

  // Step 1: Incremental backbone forward (1 token) → hidden state
  int64_t last_token = all_tokens.back();
  auto input = torch::tensor({last_token}, torch::kInt64).unsqueeze(0).to(device);
  auto hidden = model->forward_backbone(input, &kv_cache);
#ifdef __APPLE__
  if (device.is_mps()) torch::mps::synchronize();
#endif

  // hidden: [1, 1, d_model] → last position
  auto last_hidden = hidden.select(1, 0);  // [1, d_model]

  // Step 2: Main head prediction for position t+1
  auto main_logits = model->apply_lm_head(last_hidden.unsqueeze(1))
                         .squeeze(0).squeeze(0).cpu().contiguous();
  main_logits = apply_repetition_penalty(main_logits, all_tokens, repetition_penalty);
  int64_t main_token = sample_logits(main_logits, temperature, top_k, top_p, rng);

  if (main_token == eos_id) {
    all_tokens.push_back(main_token);
    return 1;
  }

  // Step 3: Draft tokens from MTP heads (just projections — no backbone forward!)
  auto draft_logits_list = model->forward_mtp_draft(last_hidden);

  std::vector<int64_t> draft_tokens;
  draft_tokens.reserve(num_drafts);

  auto temp_tokens = all_tokens;
  temp_tokens.push_back(main_token);

  for (int64_t k = 0; k < num_drafts; ++k) {
    auto dl = draft_logits_list[k].cpu().contiguous();
    dl = apply_repetition_penalty(dl, temp_tokens, repetition_penalty);
    int64_t draft_tok = sample_logits(dl, temperature, top_k, top_p, rng);
    draft_tokens.push_back(draft_tok);
    temp_tokens.push_back(draft_tok);
    if (draft_tok == eos_id) break;
  }

  total_drafted += static_cast<int64_t>(draft_tokens.size());

  // Step 4: Snapshot KV cache, then verify [main_token, draft_tokens...] in one pass
  auto snap = kv_cache.snapshot();

  // Build verification input: [main_token, draft_0, draft_1, ...]
  std::vector<int64_t> verify_seq;
  verify_seq.reserve(1 + draft_tokens.size());
  verify_seq.push_back(main_token);
  for (auto dt : draft_tokens) verify_seq.push_back(dt);

  auto verify_input = torch::tensor(
      at::IntArrayRef(verify_seq.data(), verify_seq.size()),
      torch::kInt64).unsqueeze(0).to(device);

  // This extends the KV cache with the verify sequence
  auto verify_logits = model->forward(verify_input, c10::nullopt, -100, &kv_cache);
#ifdef __APPLE__
  if (device.is_mps()) torch::mps::synchronize();
#endif
  verify_logits = verify_logits.cpu().contiguous();

  // Step 5: Accept tokens — verify_logits[0][0] predicts what comes after main_token
  // verify_logits[0][k] predicts what comes after draft_tokens[k-1]
  all_tokens.push_back(main_token);
  std::vector<uint32_t> accepted_toks = {static_cast<uint32_t>(main_token)};
  int64_t accepted = 1;
  int64_t drafts_accepted = 0;

  for (int64_t k = 0; k < static_cast<int64_t>(draft_tokens.size()); ++k) {
    if (draft_tokens[k] == eos_id) {
      all_tokens.push_back(eos_id);
      accepted++;
      drafts_accepted++;
      break;
    }

    // verify_logits[0][k] is the logits after processing verify_seq[k]
    // which predicts the token at position verify_seq[k+1]
    // So position k in verify_logits predicts what should come after draft_tokens[k-1]
    // (or after main_token when k=0)
    auto pos_logits = verify_logits.select(0, 0).select(0, k);
    int64_t model_choice = pos_logits.argmax(-1).item<int64_t>();

    if (model_choice == draft_tokens[k]) {
      all_tokens.push_back(draft_tokens[k]);
      accepted_toks.push_back(static_cast<uint32_t>(draft_tokens[k]));
      accepted++;
      drafts_accepted++;
    } else {
      // Reject: use model's choice instead
      all_tokens.push_back(model_choice);
      accepted_toks.push_back(static_cast<uint32_t>(model_choice));
      accepted++;
      break;
    }
  }

  total_accepted += drafts_accepted;

  // Rollback KV cache: keep snap + accepted tokens worth of new entries
  // snap was the length before verification. We added (1 + draft_tokens.size()) entries.
  // We want to keep snap + accepted entries.
  int64_t desired_len = snap + accepted;
  if (desired_len < kv_cache.seq_len()) {
    kv_cache.rollback(desired_len);
  }

  // Print all accepted tokens
  std::cout << tokenizer.decode(accepted_toks) << std::flush;

  return accepted;
}

}  // namespace

int main(int argc, char** argv) {
  // -----------------------------------------------------------------
  // Phase 1: parse CLI flags. Defaults below match the docblock above.
  // -----------------------------------------------------------------
  std::string checkpoint_path, config_path, vocab_path, merges_path;
  std::string device_pref = "auto";
  int64_t max_tokens = 128;
  double temperature = 0.8;
  int64_t top_k = 50;
  double top_p = 0.9;
  double repetition_penalty = 1.1;
  bool legacy_decode = false;
  bool use_kv_cache = true;
  bool use_speculative = true;
  std::string structural_config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
    else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
    else if (arg == "--vocab-file" && i + 1 < argc) vocab_path = argv[++i];
    else if (arg == "--merges-file" && i + 1 < argc) merges_path = argv[++i];
    else if (arg == "--structural-config" && i + 1 < argc) structural_config = argv[++i];
    else if (arg == "--device" && i + 1 < argc) device_pref = argv[++i];
    else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::stoll(argv[++i]);
    else if (arg == "--temperature" && i + 1 < argc) temperature = std::stod(argv[++i]);
    else if (arg == "--top-k" && i + 1 < argc) top_k = std::stoll(argv[++i]);
    else if (arg == "--top-p" && i + 1 < argc) top_p = std::stod(argv[++i]);
    else if (arg == "--repetition-penalty" && i + 1 < argc) repetition_penalty = std::stod(argv[++i]);
    else if (arg == "--legacy-decode") legacy_decode = true;
    else if (arg == "--no-kv-cache") use_kv_cache = false;
    else if (arg == "--no-speculative") use_speculative = false;
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
              << "  --no-kv-cache              (slower, avoids MPS memory issues)\n"
              << "  --no-speculative           (disable MTP speculative decoding)\n";
    return 1;
  }

  // -----------------------------------------------------------------
  // Phase 2: pick a device. "auto" prefers GPU on each platform.
  // -----------------------------------------------------------------
  if (device_pref == "auto") {
#ifdef __APPLE__
    device_pref = torch::mps::is_available() ? "mps" : "cpu";
#else
    device_pref = torch::cuda::is_available() ? "cuda" : "cpu";
#endif
  }

  auto device = select_device(device_pref);

  // Switch the IBackend implementation in olmo_cpp so model code dispatches
  // to fused CUDA kernels (GPU) or vectorized SIMD kernels (CPU).
  if (device.is_cuda()) {
    olmo_cpp::use_cuda_backend();
  } else if (device.is_cpu()) {
    olmo_cpp::use_simd_backend();
  }

  bool is_mps = device.is_mps();

  // MPS + KV cache is unstable — force no-kv-cache on MPS
  if (is_mps && use_kv_cache) {
    std::cout << "Note: KV cache disabled on MPS for stability. Using full-context mode.\n";
    use_kv_cache = false;
  }

  try {
#ifdef HAS_NLOHMANN_JSON
    // -----------------------------------------------------------------
    // Phase 3: build the model from JSON config, load weights, move to
    // the chosen device, and switch into eval mode (disables dropout).
    // -----------------------------------------------------------------
    auto cfg = olmo_cpp::load_config_from_json(config_path);
    cfg.validate();

    olmo_cpp::Transformer model(cfg);
    torch::load(model, checkpoint_path);
    model->to(device);
    model->eval();

    // Check if model has MTP heads
    bool has_mtp = model->has_mtp();
    bool do_speculative = use_speculative && has_mtp;

    if (has_mtp) {
      std::cout << "Model has " << model->num_mtp_heads()
                << " MTP heads for speculative decoding"
                << (do_speculative ? " (enabled)" : " (disabled)") << "\n";
    }

    olmo_cpp::BPETokenizer bpe_tokenizer;
    if (!bpe_tokenizer.load(vocab_path, merges_path)) {
      std::cerr << "Failed to load tokenizer\n";
      return 1;
    }
    bpe_tokenizer.set_legacy_decode(legacy_decode);

    // Optional structural tokenizer
    std::unique_ptr<olmo_cpp::StructuralTokenizer> struct_tok;
    if (!structural_config.empty()) {
      struct_tok = std::make_unique<olmo_cpp::StructuralTokenizer>();
      if (!struct_tok->load(structural_config, vocab_path, merges_path)) {
        std::cerr << "Warning: failed to load structural tokenizer, using BPE\n";
        struct_tok.reset();
      } else {
        std::cout << "Using structural tokenizer (vocab_size=" << struct_tok->vocab_size() << ")\n";
      }
    }

    // Tokenizer interface lambdas
    auto encode_text = [&](const std::string& text) -> std::vector<uint32_t> {
      if (struct_tok) {
        auto ids = struct_tok->encode(text);
        // Remove trailing EOS
        if (!ids.empty() && ids.back() == struct_tok->eos_id()) ids.pop_back();
        return ids;
      } else {
        std::vector<uint32_t> ids;
        bpe_tokenizer.encode_append(text, ids);
        return ids;
      }
    };
    auto decode_tokens = [&](const std::vector<uint32_t>& ids) -> std::string {
      if (struct_tok) return struct_tok->decode(ids);
      return bpe_tokenizer.decode(ids);
    };
    auto& tokenizer = bpe_tokenizer;  // for eos_id() access

    // -----------------------------------------------------------------
    // Phase 4: REPL loop — read prompt, generate response, repeat.
    // -----------------------------------------------------------------
    std::mt19937 rng(std::random_device{}());
    std::cout << "OLMo Chat (type 'quit' to exit)\n" << std::endl;

    while (true) {
      std::cout << "You: ";
      std::string prompt;
      if (!std::getline(std::cin, prompt)) break;
      if (prompt == "quit" || prompt == "exit" || prompt == "q") break;
      if (prompt.empty()) continue;

      std::vector<uint32_t> prompt_ids = encode_text(prompt);

      if (prompt_ids.empty()) {
        std::cout << "Model: (empty)\n" << std::endl;
        continue;
      }

      // Convert prompt to int64 tokens
      std::vector<int64_t> all_tokens(prompt_ids.begin(), prompt_ids.end());
      int64_t prompt_len = static_cast<int64_t>(all_tokens.size());
      int64_t max_total = prompt_len + max_tokens;
      if (max_total > 2048) max_total = 2048;

      std::cout << "Model: " << std::flush;

      auto gen_start = std::chrono::steady_clock::now();
      int64_t tokens_generated = 0;

      if (do_speculative) {
        // === MTP Speculative Decoding with KV Cache ===
        // Prefill: run full prompt through backbone to warm KV cache
        olmo_cpp::KVCache spec_kv(model->n_layers());
        {
          auto prefill_input = torch::tensor(
              at::IntArrayRef(all_tokens.data(), all_tokens.size()),
              torch::kInt64).unsqueeze(0).to(device);
          torch::NoGradGuard no_grad;
          model->forward_backbone(prefill_input, &spec_kv);
#ifdef __APPLE__
          if (is_mps) torch::mps::synchronize();
#endif
        }

        int64_t total_drafted = 0, total_accepted = 0;

        while (static_cast<int64_t>(all_tokens.size()) < max_total) {
          if (!all_tokens.empty() && all_tokens.back() == static_cast<int64_t>(tokenizer.eos_id()))
            break;

          int64_t accepted = speculative_decode_step(
              model, all_tokens, spec_kv, device, temperature, top_k, top_p,
              repetition_penalty, rng, tokenizer, total_drafted, total_accepted);
          tokens_generated += accepted;
        }

        auto gen_end_spec = std::chrono::steady_clock::now();
        double gen_s = std::chrono::duration<double>(gen_end_spec - gen_start).count();
        double tok_per_s = tokens_generated / (gen_s > 0 ? gen_s : 1);
        double accept_rate = total_drafted > 0 ? 100.0 * total_accepted / total_drafted : 0.0;

        std::cout << "\n[" << tokens_generated << " tokens, "
                  << std::fixed << std::setprecision(1) << tok_per_s << " tok/s, "
                  << "speculative, " << std::setprecision(0) << accept_rate << "% accepted]\n"
                  << std::endl;
        continue;  // skip the generic stats block below
      } else if (use_kv_cache) {
        // KV cache path: prefill + incremental decode
        // CUDA graphs capture the decode step (always [1,1] input) for zero launch overhead
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
          std::cout << decode_tokens(tok_to_decode) << std::flush;
          tokens_generated++;
        }
        // Note: CUDA graph capture for KV-cache decode would require static
        // KV cache tensors (pre-allocated max-length buffers). The current
        // concat-based KV cache changes shape each step, so we use direct
        // execution. For full CUDA graph support, see the paged attention
        // implementation (future work).
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
          std::cout << decode_tokens(tok_to_decode) << std::flush;
          tokens_generated++;
        }
      } else {
        // No KV cache: feed full sequence each step
        for (int64_t step = prompt_len; step < max_total; ++step) {
          int64_t last_token = all_tokens.back();
          if (last_token == static_cast<int64_t>(tokenizer.eos_id())) break;
          auto cur_seq_len = static_cast<int64_t>(all_tokens.size());
          // Clone token data to a fresh tensor (avoid dangling pointer after push_back)
          auto input_cpu = torch::tensor(
              at::IntArrayRef(all_tokens.data(), static_cast<size_t>(cur_seq_len)),
              torch::kInt64).unsqueeze(0);
          auto input = input_cpu.to(device);
          torch::NoGradGuard no_grad;
          auto logits = model->forward(input, c10::nullopt, -100, nullptr);
#ifdef __APPLE__
          if (is_mps) torch::mps::synchronize();
#endif
          // Move logits to CPU immediately to free MPS memory
          auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0).cpu().contiguous();
          // Release GPU tensors
          logits.reset();
          input.reset();
          next_logits = apply_repetition_penalty(next_logits, all_tokens, repetition_penalty);
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p, rng);
          all_tokens.push_back(next_id);
          if (next_id == static_cast<int64_t>(tokenizer.eos_id())) break;
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << decode_tokens(tok_to_decode) << std::flush;
          tokens_generated++;
        }
      }

      auto gen_end = std::chrono::steady_clock::now();
      double gen_s = std::chrono::duration<double>(gen_end - gen_start).count();
      double tok_per_s = tokens_generated / (gen_s > 0 ? gen_s : 1);

      std::cout << "\n";
      if (do_speculative) {
        std::cout << "[" << tokens_generated << " tokens, "
                  << std::fixed << std::setprecision(1) << tok_per_s << " tok/s, speculative]\n";
      } else {
        std::cout << "[" << tokens_generated << " tokens, "
                  << std::fixed << std::setprecision(1) << tok_per_s << " tok/s]\n";
      }
      std::cout << std::endl;
    }
#else
    std::cerr << "Chat requires nlohmann/json\n";
    return 1;
#endif
    return 0;
  } catch (const std::exception& e) {
    // Phase 5: surface any LibTorch / IO / config error with a clear
    // "Error:" prefix instead of letting the binary terminate via a
    // raw uncaught exception.
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
