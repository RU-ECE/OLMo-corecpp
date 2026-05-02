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

/// Bring a tensor to host via *pinned* (page-locked) memory. Pinned pages
/// skip the driver's pageable-staging copy, ~2x faster D->H transfer on
/// PCIe. Always returns a fresh writable owned buffer. On CPU/MPS this
/// degrades to .contiguous().clone() — pinning is CUDA-specific.
/// (fast-inference [12d])
torch::Tensor to_pinned_host(const torch::Tensor& t) {
  if (t.is_cuda()) {
    auto src = t.contiguous();
    auto dst = torch::empty(src.sizes(),
                            torch::TensorOptions()
                                .dtype(src.dtype())
                                .device(torch::kCPU)
                                .pinned_memory(true));
    dst.copy_(src, /*non_blocking=*/true);
    torch::cuda::synchronize();
    return dst;
  }
  return t.contiguous().clone();
}

// =====================================================================
// FAST-INFERENCE ROADMAP — beat TensorRT-LLM at one config.
// Branch: fast-inference. Target: Llama/OLMo-class 1B–7B on H100,
//   batch=1 decode latency. Numbers assume V≈50K, H100, bf16.
// =====================================================================
//
// Strategy: don't fight TRT-LLM on its home turf (general transformer
// inference, all GPUs, all batches). Specialize ruthlessly to ONE
// config and win there. ~10% from each item below; combined ~2-3x
// vs TRT-LLM at this niche is the realistic ceiling.
//
// Order is execution order, not just impact order. Earlier items
// unlock later items (e.g. paged KV unlocks CUDA graphs).
// =====================================================================
//
// PHASE 1 — UNBLOCK (weeks 1-4)
// ---------------------------------------------------------------------
//
// [1] Paged KV cache.   Prerequisite for almost everything else.
//     Current concat-based cache (chat.cpp around L655 forward call)
//     reallocates each step → shape changes → CUDA graphs impossible,
//     batching impossible, long context O(L) memcpy per step.
//     Build: fixed-size page allocator (e.g. 16 tokens/page), per-
//     request page table, kernel-side gather via page indices.
//     Reference: vLLM PagedAttention paper. ~3 weeks of focused work.
//     Files to touch: include/olmo_cpp/model/kv_cache.hpp,
//                     src/model/attention.cpp,
//                     new kernels/paged_attention.cu
//
// [2] CUDA graphs around the decode step.   ~10-30% latency.
//     Capture the [1,1]-input forward pass once, replay each step.
//     Eliminates ~500 launches × ~3-5μs of driver overhead per token.
//     Blocked on [1] (shapes must be stable).
//     Call site: tools/chat.cpp:650 (the KV-cache decode loop).
//     Reference: cudaGraphCreate / cudaGraphLaunch. ~1 week.
//
// [3] Bench harness vs TRT-LLM.   Without this you don't know if you
//     won. Pick: Llama-3-8B (or OLMo equivalent), H100 SXM, FP8/BF16,
//     128-token prompt → 256-token decode, batch=1. Measure: TPOT
//     (time per output token) and TTFT (time to first token).
//     Build: a script that runs both engines on the same prompts
//     and emits a comparison table. ~3 days.
//     Lives under: scripts/bench_vs_trtllm.sh
//
// PHASE 2 — KERNEL DOMINANCE (weeks 5-12)
// ---------------------------------------------------------------------
//
// [4] FlashAttention-2/3 decode kernel.   ~30-50% on attention.
//     Decode attention is bandwidth-bound on KV reads. Need the
//     decode variant (single Q vs many K,V) — different from the
//     training attention. Options: port FA-3, fork FlashInfer's
//     batch_decode_with_paged_kv_cache, or write fresh on top of
//     CUTLASS. ~4 weeks.
//     Reference: Tri Dao FA-3 paper, FlashInfer source.
//     Files: new kernels/decode_attention.cu, integrate via IBackend.
//
// [5] Custom LM-head GEMV.   ~20-40% on LM head step.
//     cuBLAS GEMM is tuned for square matmuls, not [1,H]·[H,V] GEMV.
//     Hand-written GEMV with split-K reduction across the V dim,
//     using TMA on H100 for W_U streaming. ~2 weeks.
//     Files: new kernels/lm_head_gemv.cu
//
// [6] Fused LM-head + sampling kernel via Gumbel-max trick.   ~5-10%.
//     sample(softmax(l/T)) ≡ argmax_i (l_i/T + g_i), g_i~Gumbel(0,1).
//     Fold into [5]: stream W_U rows, dot with hidden, generate
//     Gumbel via Philox(seed,position,vocab_idx) [or curand_uniform
//     in device API], reduce argmax, emit one int64.
//     Eliminates: [V] logits write to HBM, the multi-kernel
//     softmax/topk/sort chain, AND the 200KB D->H copy at chat.cpp
//     around L132. Phase 1: greedy + temperature only.
//     Switches std::mt19937 -> Philox; samples differ from CPU path
//     even with same seed — document this and gate behind a config flag.
//     Files: kernels/lm_head_gemv.cu (extend [5])
//     Call site: tools/chat.cpp around L545, L564, L222 (speculative).
//
// [7] Bucket-radix top-p kernel.   For when top-p is needed.
//     16 log-spaced bins, single pass histogram, scan to find cutoff
//     bucket, sort just that bucket. O(V) vs O(V log V). Min-p early
//     drop (probs < 2^-16) cuts ~70% of vocab outright.
//     Replaces CPU sort at L148-165. Fuses with [6].
//     Files: kernels/topp_radix.cu. ~1 week.
//
// PHASE 3 — QUANTIZATION (weeks 13-18)
// ---------------------------------------------------------------------
//
// [8] FP8 weight-only quantization.   ~1.5x decode throughput.
//     H100 has native FP8. Quantize weights to FP8 E4M3, keep
//     activations in BF16, dequant in the GEMM/GEMV epilogue.
//     Touches every Linear layer; biggest hit on LM head and
//     attention out_proj. Use TransformerEngine as a reference,
//     but write kernels specialized to this model. ~2-3 weeks.
//     Quality: typically <0.5% perplexity regression on calibrated
//     sets. No retraining needed.
//
// [9] INT4 weight-only quantization (AWQ-style).   ~2x on top of [8].
//     Group-quantized INT4 (g=128) with FP16 scales. Custom GEMV
//     kernel that dequants in registers. Loses ~1-2% perplexity,
//     gainable back with group-aware fine-tune. ~3-4 weeks.
//     Reference: AWQ paper, llama.cpp Q4_K_M.
//
// PHASE 4 — RESEARCH WINS (weeks 19-26)
// ---------------------------------------------------------------------
//
// [10] Speculative decoding overhaul.   1.5-3x at unchanged quality.
//      MTP path exists at speculative_decode_step (around L189) but
//      is suboptimal:
//      (a) batch the k draft sample calls (currently k separate
//          .cpu() round-trips at L322-329 — already TODO'd inline).
//      (b) tune draft length k dynamically from running accept rate.
//      (c) try a tiny separate draft model (TinyLlama 1B drafting
//          for 7B target) — typically higher acceptance than MTP.
//      (d) combine with EAGLE-2 / lookahead decoding for tree-style
//          drafting. This is publishable.
//
// [11] Persistent decode kernel.   The killer.
//      One kernel launched at startup, runs forever, polls a memory
//      queue for work. CPU writes "next token + KV ptr"; GPU runs
//      the full decode step (forward + sample) and writes the result.
//      ZERO kernel launches per token after init.
//      Saves ~500 launches × ~3μs = ~1.5ms/token of pure overhead
//      (which is huge if your forward is already optimized below 5ms).
//      Reference: TensorRT-LLM in-flight batching kernel,
//                 FlashInfer persistent kernel.
//      ~3-4 weeks. This is what gets you from "tied with TRT-LLM"
//      to "beating it."
//
// PHASE 5 — POLISH (ongoing)
// ---------------------------------------------------------------------
//
// [12] Smaller wins worth a pass:
//      - Pinned host buffers for D->H copies (cudaHostAlloc) —
//        irrelevant once [6] lands (no copies left).
//      - Fuse repetition_penalty into the sampling kernel (currently
//        a separate CPU loop at chat.cpp:105 that materializes a clone).
//      - Pre-tokenize prompt on a worker thread while model loads.
//      - bf16 end-to-end (model already supports it; verify config).
//      - Capture full conversation context once across turns instead
//        of re-encoding per turn (chat loop in main()).
//      - Replace torch::Tensor on the inference path with raw CUDA
//        pointers + a tiny shape struct — saves PyTorch dispatcher
//        overhead in C++ (~1-2μs per op, real at high token rates).
//
// EXPLICITLY NOT DOING (sunk-cost traps):
// ---------------------------------------------------------------------
//   - Rewriting from scratch. The model code, tokenizer, and existing
//     CUDA kernels (rms_norm.cu, silu_mul.cu, rope.cu) are correct
//     and reusable. Carve and replace, don't rebuild.
//   - General-purpose engine (multi-arch, multi-GPU, multi-precision).
//     Specialize to ONE target config; that's how you beat TRT-LLM.
//   - Continuous batching unless we go server-side. Single-stream
//     latency is the win condition; batching is a different game.
//   - top-p sort fusion. Use [7] (bucket-radix) instead.
//   - Standalone GPU multinomial replacement. Subsumed by [6].
//
// SUCCESS CRITERION:
// ---------------------------------------------------------------------
//   Median TPOT (time per output token) on Llama-3-8B (or equivalent),
//   H100 SXM, batch=1, 128→256 tokens, FP8: lower than TensorRT-LLM
//   v0.16+ at the same TTFT and same output quality (PPL within 1%).
//   Bench harness from [3] is the source of truth.
//
// ESTIMATED TIMELINE:
//   Single researcher full-time: 5-6 months to TRT-LLM-competitive,
//   8-10 months to clearly beating it on this niche.
//   With one collaborator: cut by ~30%.
// =====================================================================

/// Sample from logits with rep-penalty, temperature, top-k, and top-p
/// (nucleus) filtering, all fused into a single host-side preprocessing pass.
///
/// Owns one CPU clone of the logits and modifies it in place. The previous
/// `apply_repetition_penalty` step is gone — rep_tokens scatter directly into
/// this buffer, and the temperature scale is the same dense walk. One walk,
/// not two; one allocation, not two. (fast-inference [12c])
///
/// TODO(fast-inference [6]): for greedy/temperature-only (top_k<=0 &&
/// top_p>=1.0), replace this entire function with a single fused kernel
/// call inside the LM-head GEMV (roadmap item [6], depends on [5]).
/// The .cpu() below is the costliest line — 200 KB D->H/token.
/// For top-p path see roadmap item [7] (bucket-radix kernel) — keep this
/// CPU implementation only as a debug fallback once [7] lands.
int64_t sample_logits(torch::Tensor logits, double temperature,
                      int64_t top_k, double top_p,
                      const std::vector<int64_t>& rep_tokens,
                      double rep_penalty,
                      std::mt19937& gen) {
  // Bring to host via pinned memory (faster D->H on CUDA) and own the
  // resulting buffer so we can mutate it in place. (fast-inference [12d])
  auto logits_cpu = to_pinned_host(logits);
  int64_t vocab_size = logits_cpu.size(0);
  auto* p = logits_cpu.data_ptr<float>();

  // FUSED pre-pass over the host buffer:
  //   (a) rep penalty: sparse scatter into already-seen token ids
  //   (b) temperature: dense scale across the vocab
  // Order matters: rep penalty divides existing logits, so applying it
  // before temperature is mathematically equivalent to applying it after
  // (just a constant factor swap) and lets us touch the same memory once.
  if (rep_penalty != 1.0 && !rep_tokens.empty()) {
    const float pen = static_cast<float>(rep_penalty);
    for (int64_t id : rep_tokens) {
      if (id < 0 || id >= vocab_size) continue;
      const float s = p[id];
      p[id] = (s > 0.0f) ? (s / pen) : (s * pen);
    }
  }

  // Greedy: rep-penalty-modified argmax. Skip temperature/sampling pipeline.
  if (temperature <= 0.0) {
    return logits_cpu.argmax(-1).item<int64_t>();
  }

  if (temperature != 1.0) {
    const float inv_t = 1.0f / static_cast<float>(temperature);
    for (int64_t i = 0; i < vocab_size; ++i) p[i] *= inv_t;
  }

  // Top-k filtering. NOTE: this reassigns logits_cpu to a new tensor (the
  // where() output), so the `p` pointer above becomes stale here. That's
  // fine because we don't need the original buffer anymore.
  if (top_k > 0 && top_k < vocab_size) {
    auto [topk_vals, topk_indices] = logits_cpu.topk(top_k);
    auto threshold = topk_vals.index({topk_vals.size(0) - 1}).item<float>();
    logits_cpu = torch::where(logits_cpu < threshold,
                              torch::full_like(logits_cpu, -std::numeric_limits<float>::infinity()),
                              logits_cpu);
  }

  auto probs = torch::softmax(logits_cpu, -1);

  // Top-p (nucleus). TODO(fast-inference [7]): replace with bucket-radix.
  if (top_p < 1.0) {
    auto [sorted_probs, sorted_indices] = probs.sort(-1, /*descending=*/true);
    auto cumulative = sorted_probs.cumsum(-1);
    auto mask = cumulative - sorted_probs > top_p;
    sorted_probs.index_put_({mask}, 0.0f);
    probs.zero_();
    probs.scatter_(-1, sorted_indices, sorted_probs);
    auto sum = probs.sum();
    if (sum.item<float>() > 0) probs = probs / sum;
  }

  // Copy to std::vector before discrete_distribution to avoid holding tensor
  // references (fixes MPS malloc/free crashes on Apple Silicon).
  std::vector<double> probs_vec(static_cast<size_t>(vocab_size));
  auto* p_probs = probs.data_ptr<float>();
  for (int64_t i = 0; i < vocab_size; ++i) probs_vec[i] = static_cast<double>(p_probs[i]);
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

  // Step 2: Main head prediction for position t+1.
  // Skip an explicit .cpu() — sample_logits transfers via pinned memory
  // internally. (fast-inference [12d])
  auto main_logits = model->apply_lm_head(last_hidden.unsqueeze(1))
                         .squeeze(0).squeeze(0);
  int64_t main_token = sample_logits(main_logits, temperature, top_k, top_p,
                                     all_tokens, repetition_penalty, rng);

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

  // Stack all draft logits into one [num_drafts, V] tensor and do a single
  // device->host copy instead of k separate ones. Sampling itself stays
  // sequential because each step's rep_penalty depends on the previous
  // draft (temp_tokens grows per iteration). Bandwidth win: 1 sync + 1
  // transfer instead of k. (fast-inference [10a])
  auto draft_logits_stacked = torch::stack(draft_logits_list);  // [k, V] on device
  // One pinned D->H for the whole [k, V] block; subsequent loop iterations
  // view rows out of this buffer. (fast-inference [12d])
  auto draft_logits_cpu = to_pinned_host(draft_logits_stacked);

  for (int64_t k = 0; k < num_drafts; ++k) {
    auto dl = draft_logits_cpu.select(0, k);  // [V] view, no copy
    int64_t draft_tok = sample_logits(dl, temperature, top_k, top_p,
                                      temp_tokens, repetition_penalty, rng);
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
  verify_logits = to_pinned_host(verify_logits);  // (fast-inference [12d])

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

    // Tokenizer interface lambdas. encode_text_into appends into a caller-
    // owned buffer; the REPL hoists a single scratch vector out of the loop
    // so we don't reallocate every turn (BPE path skips the allocation
    // entirely; struct_tok path still allocates inside its own encode).
    // (fast-inference [12b])
    auto encode_text_into = [&](const std::string& text, std::vector<uint32_t>& out) {
      if (struct_tok) {
        auto ids = struct_tok->encode(text);
        // Remove trailing EOS
        if (!ids.empty() && ids.back() == struct_tok->eos_id()) ids.pop_back();
        out.insert(out.end(), ids.begin(), ids.end());
      } else {
        bpe_tokenizer.encode_append(text, out);
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
    std::cout << "OLMo Chat (type 'quit' to exit, 'reset' to clear context)\n" << std::endl;

    // Persistent conversation tokens — grows each turn with both the
    // user's prompt and the model's response. Each turn's prefill re-
    // processes the whole conversation (no cross-turn KV reuse yet —
    // that needs paged KV, fast-inference [1]). (fast-inference [12a])
    constexpr int64_t kMaxContext = 2048;
    std::vector<int64_t> all_tokens;
    all_tokens.reserve(static_cast<size_t>(kMaxContext));
    // Reused scratch buffer for the per-turn tokenizer output. (fast-inference [12b])
    std::vector<uint32_t> prompt_ids;
    prompt_ids.reserve(256);

    while (true) {
      std::cout << "You: ";
      std::string prompt;
      if (!std::getline(std::cin, prompt)) break;
      if (prompt == "quit" || prompt == "exit" || prompt == "q") break;
      if (prompt == "reset" || prompt == "clear") {
        all_tokens.clear();
        std::cout << "[context cleared]\n" << std::endl;
        continue;
      }
      if (prompt.empty()) continue;

      prompt_ids.clear();
      encode_text_into(prompt, prompt_ids);

      if (prompt_ids.empty()) {
        std::cout << "Model: (empty)\n" << std::endl;
        continue;
      }

      // Append new turn's tokens to running conversation.
      for (auto id : prompt_ids) {
        all_tokens.push_back(static_cast<int64_t>(id));
      }

      // Trim oldest tokens if conversation would overflow context budget.
      // Keep enough headroom for max_tokens of generation.
      int64_t budget = kMaxContext - max_tokens;
      if (budget < 1) budget = 1;
      if (static_cast<int64_t>(all_tokens.size()) > budget) {
        int64_t to_drop = static_cast<int64_t>(all_tokens.size()) - budget;
        all_tokens.erase(all_tokens.begin(), all_tokens.begin() + to_drop);
      }

      int64_t prompt_len = static_cast<int64_t>(all_tokens.size());
      int64_t max_total = prompt_len + max_tokens;
      if (max_total > kMaxContext) max_total = kMaxContext;

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
        // TODO(fast-inference [1]+[2]): this loop is THE hot path.
        //   Step 1: replace concat KV cache with paged KV ([1]) — current
        //   reshape every step blocks everything below.
        //   Step 2: cudaGraphCapture this loop body once, cudaGraphLaunch
        //   every iteration ([2]). Combined with fused sampler ([6]) and
        //   custom decode attention ([4]), this loop becomes:
        //     1 graph replay + 8B D->H per token. ~3-5x faster than today.
        //   Eventually replace the whole loop with a persistent kernel ([11]).
        olmo_cpp::KVCache kv_cache(model->n_layers());
        {
          auto input = torch::from_blob(all_tokens.data(), {1, prompt_len},
                                        torch::TensorOptions().dtype(torch::kInt64))
                           .clone()
                           .to(device);
          torch::NoGradGuard no_grad;
          auto logits = model->forward(input, c10::nullopt, -100, &kv_cache);
          auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0);
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                          all_tokens, repetition_penalty, rng);
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
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                          all_tokens, repetition_penalty, rng);
          all_tokens.push_back(next_id);
          if (next_id == static_cast<int64_t>(tokenizer.eos_id())) break;
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << decode_tokens(tok_to_decode) << std::flush;
          tokens_generated++;
        }
      } else {
        // No KV cache: feed full sequence each step
        // TODO(perf): this is the slow fallback path — O(L^2) attention recompute
        // per step. Don't optimize this loop directly; the fix is just "use the
        // KV cache path above." Keep as a correctness reference / debug mode.
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
          // Move logits to host via pinned memory (free GPU tensor right after).
          // (fast-inference [12d])
          auto next_logits = to_pinned_host(logits.select(1, logits.size(1) - 1).squeeze(0));
          // Release GPU tensors
          logits.reset();
          input.reset();
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                          all_tokens, repetition_penalty, rng);
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
