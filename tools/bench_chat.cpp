/**
 * tools/bench_chat.cpp
 *
 * Inference micro-benchmark for the fast-inference roadmap (item [3]).
 * Loads a checkpoint, runs N decode steps with greedy sampling, and emits
 * machine-readable JSON timing so two builds (baseline vs optimized) can
 * be compared apples-to-apples.
 *
 * Why this exists separately from chat.cpp:
 *   - chat.cpp is interactive and uses std::mt19937 / top-p / rep penalty.
 *     None of that is wanted in a benchmark — we want deterministic greedy
 *     output and the tightest possible decode loop.
 *   - This tool reports CUDA-event timings on GPU (microsecond precision)
 *     and std::chrono on CPU/MPS. Output is a single JSON object.
 *
 * Comparison flow:
 *   1. Build baseline (older branch) and fast-inference branch separately.
 *   2. Run both against the same checkpoint + prompt set, same decode length.
 *   3. scripts/bench_compare.sh diffs the two JSON outputs.
 *
 * Greedy only (temperature=0 path) so output tokens are deterministic and
 * the comparison is not muddied by RNG differences.
 *
 * Usage:
 *   ./build/bench_chat \
 *       --checkpoint checkpoints/1B.pt \
 *       --config configs/olmo_1B.json \
 *       --vocab-file data/gpt2/vocab.json \
 *       --merges-file data/gpt2/merges.txt \
 *       --device cuda --prompt-len 128 --decode-len 256 \
 *       --warmup 3 --iters 10 \
 *       --output results/bench_fast.json
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/paged_kv_cache.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include "olmo_cpp/backend/cuda_backend.hpp"
#include "olmo_cpp/backend/simd_backend.hpp"
#include <torch/torch.h>
#if defined(OLMO_HAS_CUDA_KERNELS) || defined(USE_CUDA)
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

#ifdef __APPLE__
#include <ATen/mps/MPSStream.h>
#endif

namespace {

torch::Device select_device(const std::string& pref) {
  if (pref == "cuda" && torch::cuda::is_available()) return torch::Device(torch::kCUDA);
#ifdef __APPLE__
  if ((pref == "mps" || pref == "metal") && torch::mps::is_available())
    return torch::Device(torch::kMPS);
#endif
  return torch::Device(torch::kCPU);
}

// Sync the device so timing measurements are valid. CUDA events handle their
// own sync; this is for MPS and chrono-based measurements.
void device_sync(torch::Device dev) {
  if (dev.is_cuda()) torch::cuda::synchronize();
#ifdef __APPLE__
  if (dev.is_mps()) torch::mps::synchronize();
#endif
}

double percentile(std::vector<double> xs, double p) {
  if (xs.empty()) return 0.0;
  std::sort(xs.begin(), xs.end());
  size_t idx = static_cast<size_t>(p * (xs.size() - 1));
  return xs[idx];
}

}  // namespace

int main(int argc, char** argv) {
  std::string checkpoint_path, config_path, vocab_path, merges_path;
  std::string int4_path;  // --int4 <sidecar.int4.pt>: INT4 weight-only inference
  std::string device_pref = "auto";
  std::string output_path = "";
  int64_t prompt_len = 128;
  int64_t decode_len = 256;
  int64_t batch = 1;
  int64_t warmup = 3;
  int64_t iters = 5;
  bool force_bf16 = false;  // cast model to BF16 for inference (tensor cores)
  // Fast decode path (CUDA, batch 1): paged KV cache + whole-step CUDA-graph
  // capture/replay. Default ON for CUDA batch-1; the eager KVCache loop below is
  // the unoptimized baseline (kept for the A/B and for MPS/CPU/batched runs).
  bool force_eager = false;      // --eager: force the eager KVCache baseline
  bool no_cuda_graph = false;    // --no-cuda-graph: paged KV but skip graph capture
  bool use_cuda_graph = false;   // resolved below for the fast path
  int64_t paged_page_size = 16;
  int64_t paged_max_seq = 2048;
  int64_t cuda_graph_warmup = 3;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--checkpoint" && i + 1 < argc) checkpoint_path = next();
    else if (a == "--config" && i + 1 < argc) config_path = next();
    else if (a == "--vocab-file" && i + 1 < argc) vocab_path = next();
    else if (a == "--merges-file" && i + 1 < argc) merges_path = next();
    else if (a == "--device" && i + 1 < argc) device_pref = next();
    else if (a == "--prompt-len" && i + 1 < argc) prompt_len = std::stoll(next());
    else if (a == "--decode-len" && i + 1 < argc) decode_len = std::stoll(next());
    else if (a == "--batch" && i + 1 < argc) batch = std::stoll(next());
    else if (a == "--warmup" && i + 1 < argc) warmup = std::stoll(next());
    else if (a == "--iters" && i + 1 < argc) iters = std::stoll(next());
    else if (a == "--output" && i + 1 < argc) output_path = next();
    else if (a == "--int4" && i + 1 < argc) int4_path = next();
    else if (a == "--bf16") force_bf16 = true;
    else if (a == "--eager") force_eager = true;
    else if (a == "--no-cuda-graph") no_cuda_graph = true;
    else if (a == "--paged-page-size" && i + 1 < argc) paged_page_size = std::stoll(next());
    else if (a == "--paged-max-seq" && i + 1 < argc) paged_max_seq = std::stoll(next());
    else if (a == "--cuda-graph-warmup" && i + 1 < argc) cuda_graph_warmup = std::stoll(next());
    else if (a == "--paged-kv" || a == "--cuda-graph") { /* accepted; fast path is default on CUDA batch-1 */ }
    else if (a == "--help" || a == "-h") {
      std::cerr << "Usage: bench_chat (--checkpoint <path> | --int4 <sidecar>) --config <path> "
                   "--vocab-file <path> --merges-file <path> "
                   "[--device cuda|mps|cpu] [--prompt-len N] [--decode-len N] "
                   "[--batch N] [--warmup N] [--iters N] [--output bench.json]\n";
      return 0;
    }
  }

  if ((checkpoint_path.empty() && int4_path.empty()) || config_path.empty() ||
      vocab_path.empty() || merges_path.empty()) {
    std::cerr << "Missing required args (need --checkpoint or --int4, plus "
                 "--config/--vocab-file/--merges-file). Use --help for usage.\n";
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

  if (device.is_cuda()) olmo_cpp::use_cuda_backend();
  else if (device.is_cpu()) olmo_cpp::use_simd_backend();

#ifdef HAS_NLOHMANN_JSON
  auto cfg = olmo_cpp::load_config_from_json(config_path);
  cfg.validate();
#else
  std::cerr << "Built without nlohmann_json — cannot parse config.\n";
  return 1;
#endif

  olmo_cpp::Transformer model(cfg);
  if (!int4_path.empty()) {
    // INT4 weight-only: load the sidecar (kept-fp params + packed int4 weights),
    // free the dense projections, then move kept-fp params to the device (the
    // int4 weights are placed on the device by enable_int4 directly).
    std::cout << "Loading INT4 sidecar: " << int4_path << "\n";
    model->enable_int4(int4_path, device);
    model->to(device);
  } else {
    // Load DIRECTLY onto the target device first (moving a CUDA-saved checkpoint
    // through CPU can corrupt base-weight storage). bf16-trained checkpoints store
    // BF16 weights: an fp32 model mismatches storage size on load, so try fp32
    // then fall back to a BF16 model.
    model->to(device);
    try {
      torch::load(model, checkpoint_path, device);
    } catch (const c10::Error&) {
      // bf16-saved checkpoint: load on CPU, cast to fp32 on the HOST, THEN move to
      // device. Moving a bf16 module to CUDA first segfaults in the H2D copy on
      // this box's CUDA 13; upcasting to fp32 host-side first sidesteps it.
      model = olmo_cpp::Transformer(cfg);
      model->to(torch::kBFloat16);
      torch::load(model, checkpoint_path, torch::kCPU);
      model->to(torch::kFloat32);
      model->to(device);
    }
  }
  // --bf16 casts AFTER the model is on the device, so the (unstable) bf16 host->
  // device copy never happens; the cast runs on-device where it is safe.
  if (force_bf16) model->to(torch::kBFloat16);
  model->to(device);
  model->eval();
  torch::NoGradGuard no_grad;

  olmo_cpp::BPETokenizer tok;
  if (!tok.load(vocab_path, merges_path)) {
    std::cerr << "Tokenizer load failed.\n";
    return 1;
  }

  // Build a fixed deterministic prompt. Real content doesn't matter for
  // benchmarking — only shape and decode length do. Use the EOS token id
  // (GPT-2 reuses 50256 as BOS) as the leading token; rest are token 0.
  std::vector<int64_t> prompt(prompt_len, 0);
  prompt[0] = static_cast<int64_t>(tok.eos_id());

  // Per-iter measurements.
  std::vector<double> ttft_ms;     // time-to-first-token (prefill)
  std::vector<double> tpot_ms;     // mean time-per-output-token
  std::vector<double> total_ms;    // end-to-end one full decode
  std::vector<double> all_step_ms; // every per-step timing across iters (for p50/p99)

  auto run_once = [&](bool measure) -> void {
    olmo_cpp::KVCache kv(model->n_layers());

    // Build [B, prompt_len] input by tiling the same prompt across batch.
    auto input = torch::tensor(prompt, torch::kInt64).unsqueeze(0)  // [1, T]
                     .repeat({batch, 1}).to(device);                // [B, T]

    auto t_start = std::chrono::steady_clock::now();
    device_sync(device);

    // Prefill
    auto logits = model->forward(input, c10::nullopt, -100, &kv);
    device_sync(device);
    auto t_prefill = std::chrono::steady_clock::now();

    // Greedy: argmax of last position
    auto next_tok = logits.select(1, logits.size(1) - 1).argmax(-1);  // [B]

    // Decode loop
    std::vector<double> step_ms;
    step_ms.reserve(static_cast<size_t>(decode_len));
    for (int64_t s = 0; s < decode_len; ++s) {
      auto step_start = std::chrono::steady_clock::now();
      auto step_input = next_tok.unsqueeze(1);  // [B, 1]
      auto out = model->forward(step_input, c10::nullopt, -100, &kv);
      next_tok = out.select(1, 0).argmax(-1);   // [B]
      device_sync(device);
      auto step_end = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double, std::milli>(step_end - step_start).count();
      step_ms.push_back(dt);
    }

    auto t_end = std::chrono::steady_clock::now();

    if (measure) {
      double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
      double decode_total_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill).count();
      ttft_ms.push_back(prefill_ms);
      tpot_ms.push_back(decode_total_ms / static_cast<double>(decode_len));
      total_ms.push_back(prefill_ms + decode_total_ms);
      for (auto v : step_ms) all_step_ms.push_back(v);
    }
  };

  // ---- Fast path (CUDA, batch 1): paged KV + CUDA-graph capture/replay ----
  // Mirrors tools/chat.cpp's optimized decode: prefill through forward_paged,
  // a few eager warmup steps (allocator settle), then capture the whole decode
  // step into one CUDA graph and replay it — eliminating per-step kernel-launch
  // overhead (the reason eager decode is overhead-bound, ~15 tok/s, on CUDA).
  const bool fast = device.is_cuda() && batch == 1 && !force_eager;
  if (fast) {
    use_cuda_graph = !no_cuda_graph;
    // The graph-safe paged-KV write kernel requires FP32 pools; bf16/fp16 + CUDA
    // graph is unsupported (paged_attention.cu TORCH_CHECK "pools must be
    // float32"). Upcast a bf16/fp16 model to fp32 so the fast path runs the
    // documented fp32 + cuda-graph config instead of crashing. (INT4 keeps its
    // fp32 kept-params, so its pools are already fp32 — left untouched.)
    if (use_cuda_graph && !model->parameters().empty()) {
      auto dt = model->parameters()[0].dtype().toScalarType();
      if (dt == torch::kBFloat16 || dt == torch::kHalf) {
        std::cerr << "[bench] cuda-graph fast path needs fp32 pools; upcasting "
                     "model to fp32 (bf16 + cuda-graph is unsupported).\n";
        model->to(torch::kFloat32);
      }
    }
  }

#if defined(OLMO_HAS_CUDA_KERNELS) || defined(USE_CUDA)
  auto run_once_paged = [&](bool measure) -> void {
    torch::NoGradGuard no_grad;
    const int64_t n_kv_heads = cfg.get_n_kv_heads();
    const int64_t head_dim   = cfg.get_head_dim();
    const int64_t max_pages  = (paged_max_seq + paged_page_size - 1) / paged_page_size;
    auto model_dtype = torch::kFloat32;
    if (!model->parameters().empty())
      model_dtype = model->parameters()[0].dtype().toScalarType();
    const bool graph_mode = use_cuda_graph && device.is_cuda();
    auto paged = graph_mode
        ? olmo_cpp::make_paged_kv_cache_graph_safe(model->n_layers(), n_kv_heads,
              head_dim, paged_page_size, max_pages, device, model_dtype)
        : olmo_cpp::make_paged_kv_cache(model->n_layers(), n_kv_heads,
              head_dim, paged_page_size, max_pages, device, model_dtype);

    // Prefill [1, prompt_len].
    auto t_start = std::chrono::steady_clock::now();
    device_sync(device);
    auto pin = torch::tensor(prompt, torch::kInt64).unsqueeze(0).to(device);  // [1, T]
    auto logits = model->forward_paged(pin, paged.get());
    int64_t next_tok = logits.select(1, logits.size(1) - 1).argmax(-1).item<int64_t>();
    device_sync(device);
    auto t_prefill = std::chrono::steady_clock::now();

    std::vector<double> step_ms;
    step_ms.reserve(static_cast<size_t>(decode_len));
    int64_t produced = 0;

    // Reused [1,1] input buffer (no per-step host->device alloc).
    auto step_buf = torch::empty({1, 1},
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto eager_step = [&](int64_t last) -> int64_t {
      auto step_start = std::chrono::steady_clock::now();
      step_buf.fill_(last);
      auto out = model->forward_paged(step_buf, paged.get());
      int64_t nt = out.select(1, out.size(1) - 1).squeeze(0).argmax(-1).item<int64_t>();
      device_sync(device);
      auto step_end = std::chrono::steady_clock::now();
      step_ms.push_back(std::chrono::duration<double, std::milli>(step_end - step_start).count());
      return nt;
    };

    // Eager warmup steps before capture (only meaningful in graph mode).
    const int64_t warm = graph_mode ? cuda_graph_warmup : 0;
    for (int64_t s = 0; s < warm && produced < decode_len; ++s) {
      next_tok = eager_step(next_tok);
      ++produced;
    }

    if (graph_mode && produced < decode_len) {
      auto static_input = torch::empty({1, 1},
          torch::TensorOptions().dtype(torch::kInt64).device(device));
      static_input.fill_(next_tok);
      paged->set_external_advance(true);   // we bump the cursor, not append()
      paged->advance_cursor(1);            // slot for the to-be-captured step
      auto cap_stream = c10::cuda::getStreamFromPool();
      c10::cuda::CUDAStreamGuard guard(cap_stream);
      at::cuda::CUDAGraph graph;
      torch::Tensor captured_logits;
      {  // capture run == first real decode step under graph mode
        auto step_start = std::chrono::steady_clock::now();
        graph.capture_begin();
        captured_logits = model->forward_paged(static_input, paged.get());
        graph.capture_end();
        next_tok = captured_logits.select(1, 0).squeeze(0).argmax(-1).item<int64_t>();
        device_sync(device);
        auto step_end = std::chrono::steady_clock::now();
        step_ms.push_back(std::chrono::duration<double, std::milli>(step_end - step_start).count());
        ++produced;
      }
      while (produced < decode_len) {  // replay loop — one launch per token
        auto step_start = std::chrono::steady_clock::now();
        static_input.fill_(next_tok);
        paged->advance_cursor(1);
        graph.replay();
        next_tok = captured_logits.select(1, 0).squeeze(0).argmax(-1).item<int64_t>();
        device_sync(device);
        auto step_end = std::chrono::steady_clock::now();
        step_ms.push_back(std::chrono::duration<double, std::milli>(step_end - step_start).count());
        ++produced;
      }
    } else {
      while (produced < decode_len) {  // paged eager fallback (--no-cuda-graph)
        next_tok = eager_step(next_tok);
        ++produced;
      }
    }

    auto t_end = std::chrono::steady_clock::now();
    if (measure) {
      double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
      double decode_total_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill).count();
      ttft_ms.push_back(prefill_ms);
      tpot_ms.push_back(decode_total_ms / static_cast<double>(decode_len));
      total_ms.push_back(prefill_ms + decode_total_ms);
      for (auto v : step_ms) all_step_ms.push_back(v);
    }
  };
#endif

  auto do_run = [&](bool measure) {
#if defined(OLMO_HAS_CUDA_KERNELS) || defined(USE_CUDA)
    if (fast) { run_once_paged(measure); return; }
#endif
    run_once(measure);
  };

  for (int64_t w = 0; w < warmup; ++w) do_run(false);
  for (int64_t i = 0; i < iters; ++i) do_run(true);

  auto mean = [](const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  };

  double ttft_mean = mean(ttft_ms);
  double tpot_mean = mean(tpot_ms);
  double total_mean = mean(total_ms);
  double tpot_p50 = percentile(all_step_ms, 0.50);
  double tpot_p99 = percentile(all_step_ms, 0.99);
  double tok_per_s = 1000.0 * batch / tpot_mean;

  const char* path = fast ? (use_cuda_graph ? "paged-kv+cuda-graph" : "paged-kv-eager")
                          : "eager-kvcache";
  std::cerr << "device=" << device << " batch=" << batch << " path=" << path
            << " prompt_len=" << prompt_len << " decode_len=" << decode_len << "\n"
            << "TTFT mean: " << ttft_mean << " ms\n"
            << "TPOT mean: " << tpot_mean << " ms (p50=" << tpot_p50
            << ", p99=" << tpot_p99 << ")\n"
            << "Throughput: " << tok_per_s << " tok/s (batch=" << batch << ")\n";

#ifdef HAS_NLOHMANN_JSON
  if (!output_path.empty()) {
    nlohmann::json j;
    j["device"] = device_pref;
    j["batch"] = batch;
    j["prompt_len"] = prompt_len;
    j["decode_len"] = decode_len;
    j["warmup"] = warmup;
    j["iters"] = iters;
    j["ttft_ms_mean"] = ttft_mean;
    j["tpot_ms_mean"] = tpot_mean;
    j["tpot_ms_p50"] = tpot_p50;
    j["tpot_ms_p99"] = tpot_p99;
    j["total_ms_mean"] = total_mean;
    j["throughput_tok_per_s"] = tok_per_s;
    j["ttft_ms_all"] = ttft_ms;
    j["tpot_ms_all"] = tpot_ms;
    std::ofstream out(output_path);
    out << j.dump(2) << "\n";
    std::cerr << "Wrote " << output_path << "\n";
  }
#endif

  return 0;
}
