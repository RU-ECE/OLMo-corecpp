/**
 * tools/bench_spec_decode.cpp — real decode speed with SL-MTP self-speculation.
 *
 * Loads a base checkpoint (no MTP heads), attaches SL-MTP heads trained by
 * `slmtp_killtest train --save`, and decodes the same prompts twice:
 *   plain   one token per forward (spec::plain_generate)
 *   spec    spec::mtp_prefill + spec::mtp_step: one forward per step,
 *           drafts from the heads with the main logits reused (no W read)
 * Greedy, so both must produce identical tokens; the tool checks that and
 * reports tokens/s, the speedup, acceptance and tokens per forward.
 *
 *   bench_spec_decode --config base.json --ckpt base.pt --heads heads.pt --rank 128
 *       --vocab v.json --merges m.txt --text stories.txt [--K 3] [--draft 3]
 *       [--prompts 8] [--prompt-len 64] [--new 128] [--skip 1000] [--device mps]
 */
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include "olmo_cpp/generate/spec_core.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/sl_mtp_head.hpp"
#include "olmo_cpp/model/transformer.hpp"

#include <torch/torch.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace olmo_cpp;

namespace {

std::map<std::string, std::string> parse(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) continue;
    k = k.substr(2);
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a[k] = argv[++i];
    else a[k] = "1";
  }
  return a;
}

std::string get(const std::map<std::string, std::string>& a, const std::string& k, const std::string& d = "") {
  auto it = a.find(k);
  if (it != a.end()) return it->second;
  TORCH_CHECK(!d.empty(), "missing --", k);
  return d;
}

void sync(const torch::Device& d) {
  if (d.is_mps()) torch::mps::synchronize();
  if (d.is_cuda()) torch::cuda::synchronize();
}

}  // namespace

int main(int argc, char** argv) {
  auto a = parse(argc, argv);
  const std::string dev_s = get(a, "device", "mps");
  torch::Device dev = dev_s == "cuda" ? torch::Device(torch::kCUDA)
                    : dev_s == "cpu"  ? torch::Device(torch::kCPU)
                                      : torch::Device(torch::kMPS);
  const int64_t K = std::stoll(get(a, "K", "3"));
  const int64_t draft = std::stoll(get(a, "draft", std::to_string(K)));
  const int64_t r = std::stoll(get(a, "rank"));
  const int64_t n_prompts = std::stoll(get(a, "prompts", "8"));
  const int64_t prompt_len = std::stoll(get(a, "prompt-len", "64"));
  const int64_t n_new = std::stoll(get(a, "new", "128"));
  const int64_t skip = std::stoll(get(a, "skip", "1000"));  // stories to skip (past the training text)
  const int64_t fanout = std::stoll(get(a, "tree", "0"));    // > 0: tree speculation with this fanout

  // Base weights into a model that also has SL-MTP heads.
  auto base_cfg = load_config_from_json(get(a, "config"));
  base_cfg.validate();
  auto cfg = base_cfg;
  cfg.num_mtp_heads = K;
  cfg.mtp_head_type = "shared_lowrank";
  cfg.mtp_rank = r;
  Transformer model(cfg);
  {
    Transformer base(base_cfg);
    torch::load(base, get(a, "ckpt"), torch::kCPU);
    torch::NoGradGuard ng;
    auto dst = model->named_parameters();
    int64_t copied = 0;
    for (auto& it : base->named_parameters()) {
      TORCH_CHECK(dst.contains(it.key()), "no parameter ", it.key(), " in the MTP model");
      dst[it.key()].copy_(it.value());
      ++copied;
    }
    // Copy buffers too (not just parameters): anything the checkpoint carries
    // outside named_parameters must match, or decoding silently changes.
    auto dbuf = model->named_buffers();
    int64_t nbuf = 0, nbuf_diff = 0;
    for (auto& it : base->named_buffers()) {
      if (!dbuf.contains(it.key())) continue;
      ++nbuf;
      if (!dbuf[it.key()].sizes().equals(it.value().sizes()) || !torch::equal(dbuf[it.key()], it.value())) ++nbuf_diff;
      dbuf[it.key()].copy_(it.value());
    }
    std::fprintf(stderr, "buffers: %lld matched, %lld differed from a fresh model (copied)\n", (long long)nbuf,
                 (long long)nbuf_diff);
    if (a.count("debug-base")) {  // the base model's own prediction on a fixed input
      auto x = torch::arange(100, 164, torch::kInt64).unsqueeze(0);
      base->eval();
      auto lb = base->forward(x).select(1, 63).reshape({-1});
      auto lm = model->forward(x).select(1, 63).reshape({-1});
      std::fprintf(stderr, "debug: base vs copied model last-position logits max |diff| %.3e\n",
                   (lb - lm).abs().max().item<double>());
    }
    std::vector<torch::Tensor> heads;
    torch::load(heads, get(a, "heads"));
    TORCH_CHECK(static_cast<int64_t>(heads.size()) == 3 * K, "expected ", 3 * K, " head tensors, got ", heads.size());
    for (int64_t k = 0; k < K; ++k) {
      const std::string pre = "mtp_heads." + std::to_string(k) + ".";
      dst[pre + "U"].copy_(heads[3 * k]);
      dst[pre + "V"].copy_(heads[3 * k + 1]);
      dst[pre + "a"].copy_(heads[3 * k + 2]);
    }
    std::fprintf(stderr, "loaded %lld base tensors + %lld SL-MTP heads (rank %lld)\n", (long long)copied,
                 (long long)K, (long long)r);
  }
  model->to(dev);
  model->eval();

  // Prompts: the first prompt_len tokens of stories after the first `skip`.
  BPETokenizer tok;
  TORCH_CHECK(tok.load(get(a, "vocab"), get(a, "merges")), "tokenizer load failed");
  std::ifstream tf(get(a, "text"));
  std::stringstream ss;
  ss << tf.rdbuf();
  const std::string text = ss.str(), sep = "<|endoftext|>";
  std::vector<std::vector<int64_t>> prompts;
  size_t pos = 0;
  int64_t story = 0;
  while (pos < text.size() && static_cast<int64_t>(prompts.size()) < n_prompts) {
    size_t e = text.find(sep, pos);
    if (e == std::string::npos) e = text.size();
    if (story++ >= skip) {
      auto ids = tok.encode(text.substr(pos, e - pos));
      if (static_cast<int64_t>(ids.size()) > prompt_len + 8)
        prompts.emplace_back(ids.begin(), ids.begin() + prompt_len);
    }
    pos = e + sep.size();
  }
  const int64_t eos = static_cast<int64_t>(tok.eos_id());

  // Sanity: on THIS model and device, a cached multi-token block must give the
  // same logits as the no-cache forward (what the verify pass relies on).
  {
    torch::NoGradGuard ng;
    const auto& t = prompts[0];
    auto all = torch::tensor(at::IntArrayRef(t.data(), t.size()), torch::kInt64).unsqueeze(0).to(dev);
    auto full = model->forward(all).to(torch::kCPU);
    KVCache kv(cfg.n_layers, dev);
    const int64_t P = static_cast<int64_t>(t.size()) - 4;
    model->forward(all.narrow(1, 0, P), c10::nullopt, -100, &kv);
    auto blk = model->forward(all.narrow(1, P, 4), c10::nullopt, -100, &kv).to(torch::kCPU);
    auto ref = full.narrow(1, P, 4);
    {
      KVCache kv2(cfg.n_layers, dev);
      auto hp = model->forward_backbone(all, &kv2);
      auto lp = model->apply_lm_head(hp.select(1, hp.size(1) - 1).unsqueeze(1).contiguous()).reshape({-1});
      auto lf = full.select(1, full.size(1) - 1).reshape({-1});
      std::vector<uint32_t> a1 = {static_cast<uint32_t>(lf.argmax().item<int64_t>())},
                            a2 = {static_cast<uint32_t>(lp.to(torch::kCPU).argmax().item<int64_t>())};
      std::fprintf(stderr, "next token after the prompt: no-cache forward '%s', cached prefill '%s'\n",
                   tok.decode(a1).c_str(), tok.decode(a2).c_str());
    }
    std::fprintf(stderr, "cache parity on the real model: max |diff| %.3e, argmax equal %s\n",
                 (blk - ref).abs().max().item<double>(),
                 torch::equal(blk.argmax(-1), ref.argmax(-1)) ? "yes" : "NO");
  }

  spec::Sampling greedy;
  std::mt19937 rng(0);
  // Warm up both paths once (kernel compilation, allocator).
  spec::plain_generate(model, prompts[0], 8, greedy, rng, eos);
  {
    spec::MtpState st(cfg.n_layers, dev);
    auto h = prompts[0];
    spec::mtp_prefill(model, st, h, greedy, rng, eos);
    for (int i = 0; i < 3; ++i) spec::mtp_step(model, st, h, draft, greedy, rng, eos);
  }

  double t_plain = 0, t_spec = 0;
  int64_t tok_plain = 0, tok_spec = 0, drafted = 0, accepted = 0, steps = 0, mismatches = 0;
  for (const auto& p : prompts) {
    sync(dev);
    auto t0 = std::chrono::steady_clock::now();
    auto ref = spec::plain_generate(model, p, n_new, greedy, rng, eos);
    sync(dev);
    auto t1 = std::chrono::steady_clock::now();
    spec::MtpState st(cfg.n_layers, dev);
    auto h = p;
    spec::mtp_prefill(model, st, h, greedy, rng, eos);
    while (!st.done && static_cast<int64_t>(h.size() - p.size()) < n_new) {
      if (fanout > 0) spec::mtp_tree_step(model, st, h, fanout, draft, greedy, eos);
      else spec::mtp_step(model, st, h, draft, greedy, rng, eos);
    }
    sync(dev);
    auto t2 = std::chrono::steady_clock::now();
    std::vector<int64_t> got(h.begin() + static_cast<long>(p.size()), h.end());
    got.resize(std::min(got.size(), ref.size()));
    if (!std::equal(got.begin(), got.end(), ref.begin())) {
      size_t i = 0;
      while (i < got.size() && got[i] == ref[i]) ++i;
      if (mismatches == 0) {
        // Was it a near-tie? Recompute the plain path's logits at that position.
        torch::NoGradGuard ng;
        std::vector<int64_t> ctx(p.begin(), p.end());
        ctx.insert(ctx.end(), ref.begin(), ref.begin() + static_cast<long>(i));
        auto lg = model->forward(torch::tensor(at::IntArrayRef(ctx.data(), ctx.size()), torch::kInt64)
                                     .unsqueeze(0).to(dev))
                      .select(1, static_cast<int64_t>(ctx.size()) - 1).reshape({-1}).to(torch::kCPU);
        auto top = std::get<0>(lg.topk(2));
        std::fprintf(stderr, "first divergence at new token %zu: spec %lld vs plain %lld; top-2 logit gap there %.2e\n",
                     i, (long long)got[i], (long long)ref[i], (top[0] - top[1]).item<double>());
      }
      ++mismatches;
    }
    if (a.count("show") && &p == &prompts[0]) {
      std::vector<uint32_t> pr(p.begin(), p.end()), gen(ref.begin(), ref.end());
      std::fprintf(stderr, "--- prompt: %s\n--- greedy: %s\n", tok.decode(pr).c_str(), tok.decode(gen).c_str());
    }
    t_plain += std::chrono::duration<double>(t1 - t0).count();
    t_spec += std::chrono::duration<double>(t2 - t1).count();
    tok_plain += static_cast<int64_t>(ref.size());
    tok_spec += static_cast<int64_t>(h.size() - p.size());
    drafted += st.drafted;
    accepted += st.accepted;
    steps += st.steps + 1;  // + the prefill forward
  }
  const double plain_tps = tok_plain / t_plain, spec_tps = tok_spec / t_spec;
  std::printf("{\"device\": \"%s\", \"tree_fanout\": %lld, \"rank\": %lld, \"heads\": %lld, \"draft\": %lld, \"prompts\": %zu, "
              "\"new_tokens\": %lld, \"plain_tok_s\": %.2f, \"spec_tok_s\": %.2f, \"speedup\": %.3f, "
              "\"acceptance\": %.4f, \"tokens_per_forward\": %.3f, \"outputs_identical\": %s}\n",
              dev_s.c_str(), (long long)fanout, (long long)r, (long long)K, (long long)draft, prompts.size(), (long long)n_new,
              plain_tps, spec_tps, spec_tps / plain_tps, drafted ? double(accepted) / drafted : 0.0,
              double(tok_spec) / steps, mismatches == 0 ? "true" : "false");
  return mismatches == 0 ? 0 : 1;
}
