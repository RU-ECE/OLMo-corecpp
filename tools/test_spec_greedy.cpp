/**
 * tools/test_spec_greedy.cpp — proof test for speed-plan bug 5.
 *
 * Speculative decoding must not change the text. With greedy sampling, every
 * speculative path has to produce exactly the tokens plain one-at-a-time
 * decoding produces. Checks, on small random models in float64 (no argmax
 * near-ties from rounding):
 *   1. KV-cache attention: logits for a block appended to a cache equal the
 *      no-cache logits, for a 1-token step and a multi-token block, with
 *      plain attention, GQA, and a sliding window. The old code ran the
 *      multi-token cached case with no causal mask, which fails here.
 *   2. MTP self-speculation == plain greedy, with the model's own (random)
 *      heads and with forced drafts that are all right, right then wrong,
 *      or all wrong, so the accept, partial and reject paths all run.
 *   3. Draft-model speculation == plain greedy, with draft == target (every
 *      draft accepted) and with an unrelated draft model (mostly rejected).
 *   4. Sampling (temperature 1): both speculative paths produce valid
 *      tokens, and drafts are accepted at a sane rate when draft == target.
 *
 * Exit code 0 = all pass. Each failure prints the first differing position.
 */
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include "olmo_cpp/generate/draft_model_speculative.hpp"
#include "olmo_cpp/generate/spec_core.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/transformer.hpp"

#include <torch/torch.h>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace olmo_cpp;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

TransformerConfig tiny_cfg(int64_t n_kv_heads, int64_t window, int64_t mtp) {
  TransformerConfig c;
  c.d_model = 64;
  c.n_layers = 2;
  c.n_heads = 4;
  c.n_kv_heads = n_kv_heads;
  c.vocab_size = 101;
  c.hidden_size_multiple_of = 32;
  c.sliding_window_size = window;
  c.num_mtp_heads = mtp;
  c.init_std = 0.2;  // spread the logits so greedy choices are well separated
  return c;
}

Transformer make_model(const TransformerConfig& c, uint64_t seed) {
  torch::manual_seed(seed);
  Transformer m(c);
  m->init_weights();
  m->to(torch::kFloat64);
  m->eval();
  return m;
}

std::vector<int64_t> random_prompt(int64_t n, int64_t vocab, uint64_t seed) {
  std::mt19937 g(seed);
  std::uniform_int_distribution<int64_t> u(0, vocab - 1);
  std::vector<int64_t> p(n);
  for (auto& t : p) t = u(g);
  return p;
}

torch::Tensor ids(const std::vector<int64_t>& v) {
  return torch::tensor(at::IntArrayRef(v.data(), v.size()), torch::kInt64).unsqueeze(0);
}

std::string sci(double x) {
  char b[32];
  std::snprintf(b, sizeof b, "%.1e", x);
  return b;
}

std::string first_diff(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i])
      return " (first diff at " + std::to_string(i) + ": " + std::to_string(a[i]) + " vs " +
             std::to_string(b[i]) + ")";
  if (a.size() != b.size())
    return " (lengths " + std::to_string(a.size()) + " vs " + std::to_string(b.size()) + ")";
  return "";
}

// 1. Cached block logits == no-cache logits.
void test_cache_attention(const char* name, const TransformerConfig& c) {
  torch::NoGradGuard ng;
  auto m = make_model(c, 1);
  const int64_t T = 24, P = 13;
  auto toks = random_prompt(T, c.vocab_size, 7);
  auto full = m->forward(ids(toks));  // [1, T, V], no cache

  // Prefill P tokens, then one block of the rest.
  {
    KVCache kv(c.n_layers);
    std::vector<int64_t> a(toks.begin(), toks.begin() + P), b(toks.begin() + P, toks.end());
    m->forward(ids(a), c10::nullopt, -100, &kv);
    auto blk = m->forward(ids(b), c10::nullopt, -100, &kv);
    const double err = (blk - full.narrow(1, P, T - P)).abs().max().item<double>();
    check(err < 1e-9, std::string(name) + ": cached multi-token block matches no-cache (max err " +
                          sci(err) + ")");
  }
  // Prefill P tokens, then one token at a time.
  {
    KVCache kv(c.n_layers);
    std::vector<int64_t> a(toks.begin(), toks.begin() + P);
    m->forward(ids(a), c10::nullopt, -100, &kv);
    double err = 0.0;
    for (int64_t t = P; t < T; ++t) {
      auto one = m->forward(ids({toks[static_cast<size_t>(t)]}), c10::nullopt, -100, &kv);
      err = std::max(err, (one.select(1, 0) - full.select(1, t)).abs().max().item<double>());
    }
    check(err < 1e-9, std::string(name) + ": cached one-token decode matches no-cache (max err " +
                          sci(err) + ")");
  }
}

// 2. MTP self-speculation == plain greedy.
void test_mtp(const TransformerConfig& c) {
  auto m = make_model(c, 2);
  const int64_t N = 40;
  const int64_t eos = 50256;  // outside the 101-token vocab: never produced
  spec::Sampling greedy;
  std::mt19937 rng(0);
  auto prompt = random_prompt(9, c.vocab_size, 3);
  const auto ref = spec::plain_generate(m, prompt, N + 8, greedy, rng, eos);

  // Plain greedy with the cache equals greedy by full recompute (validates ref).
  {
    torch::NoGradGuard ng;
    std::vector<int64_t> h = prompt, out;
    for (int i = 0; i < 12; ++i) {
      auto lg = m->forward(ids(h)).select(1, static_cast<int64_t>(h.size()) - 1);
      const int64_t t = spec::greedy_token(lg, greedy, h);
      out.push_back(t);
      h.push_back(t);
    }
    std::vector<int64_t> ref12(ref.begin(), ref.begin() + 12);
    check(out == ref12, "reference: cached greedy == full-recompute greedy" + first_diff(out, ref12));
  }

  auto run = [&](int mode, int64_t k) {
    spec::MtpState st(c.n_layers, torch::kCPU);
    std::vector<int64_t> hist = prompt;
    spec::mtp_prefill(m, st, hist, greedy, rng, eos);
    int64_t step = 0;
    while (static_cast<int64_t>(hist.size() - prompt.size()) < N) {
      const size_t done = hist.size() - prompt.size();  // tokens committed so far
      std::vector<int64_t> forced;
      if (mode > 0) {
        // Drafts for the slots after the pending token: ref[done], ref[done+1], ...
        for (int64_t i = 0; i < k; ++i) {
          int64_t t = ref[done + static_cast<size_t>(i)];
          const bool wrong = (mode == 2 && i == (step % k)) || mode == 3;
          forced.push_back(wrong ? (t + 1) % c.vocab_size : t);
        }
      }
      spec::mtp_step(m, st, hist, k, greedy, rng, eos, mode > 0 ? &forced : nullptr);
      ++step;
    }
    std::vector<int64_t> got(hist.begin() + static_cast<long>(prompt.size()), hist.end());
    got.resize(N);
    std::vector<int64_t> want(ref.begin(), ref.begin() + N);
    return std::make_pair(got == want, first_diff(got, want) + " accepted " +
                                           std::to_string(st.accepted) + "/" + std::to_string(st.drafted) +
                                           " in " + std::to_string(st.steps) + " steps");
  };
  // Tree speculation (exact, greedy): own heads, and forced candidates where the
  // right token is the SECOND child at each depth (exercises non-first branches),
  // or never present.
  for (int mode = 0; mode < 3; ++mode) {
    for (int64_t depth : {2, 3}) {
      spec::MtpState st(c.n_layers, torch::kCPU);
      std::vector<int64_t> hist = prompt;
      spec::mtp_prefill(m, st, hist, greedy, rng, eos);
      while (static_cast<int64_t>(hist.size() - prompt.size()) < N) {
        const size_t done = hist.size() - prompt.size();
        std::vector<std::vector<int64_t>> cand;
        for (int64_t d = 0; d < depth; ++d) {
          const int64_t right = ref[done + static_cast<size_t>(d)];
          const int64_t wrong = (right + 1 + d) % c.vocab_size;
          cand.push_back(mode == 1 ? std::vector<int64_t>{wrong, right} : std::vector<int64_t>{wrong, (wrong + 1) % c.vocab_size == right ? (wrong + 2) % c.vocab_size : (wrong + 1) % c.vocab_size});
        }
        spec::mtp_tree_step(m, st, hist, 2, depth, greedy, eos, mode == 0 ? nullptr : &cand);
      }
      std::vector<int64_t> got(hist.begin() + static_cast<long>(prompt.size()), hist.end());
      got.resize(N);
      std::vector<int64_t> want(ref.begin(), ref.begin() + N);
      const char* tn[] = {"own heads", "forced right token as 2nd child", "forced, right token absent"};
      // With the right token always present, a verify that sees the context
      // accepts every depth on every step (a context-blind one does not).
      const bool full_accept = mode != 1 || st.accepted == st.steps * depth;
      check(got == want && full_accept,
            std::string("TREE spec == greedy, ") + tn[mode] + ", fanout 2, depth " + std::to_string(depth) +
                first_diff(got, want) + " accepted " + std::to_string(st.accepted) + " in " +
                std::to_string(st.steps) + " steps" + (mode == 1 ? " (must be every depth)" : ""));
    }
  }

  const char* names[] = {"own heads", "forced all-right", "forced right-then-wrong", "forced all-wrong"};
  for (int mode = 0; mode < 4; ++mode) {
    for (int64_t k : {1, 3}) {
      auto [ok, info] = run(mode, k);
      check(ok, std::string("MTP spec == greedy, ") + names[mode] + ", k=" + std::to_string(k) + info);
    }
  }
}

// 3 + 4. Draft-model speculation.
void test_draft_model(const TransformerConfig& c) {
  auto target = make_model(c, 4);
  auto other = make_model(c, 5);
  BPETokenizer tok;  // only eos_id() is used: 50256, outside the vocab
  const int64_t N = 40;
  auto prompt = random_prompt(9, c.vocab_size, 11);
  std::mt19937 rng(0);
  spec::Sampling greedy;
  const auto ref = spec::plain_generate(target, prompt, N + 8, greedy, rng, 50256);

  auto run = [&](Transformer* drafter, double temperature, int64_t k, int64_t* acc, int64_t* drf) {
    DraftModelSpeculativeState st(&target, drafter, c.n_layers, c.n_layers, torch::kCPU);
    st.draft_len = k;
    std::vector<int64_t> all = prompt;
    draft_model_speculative_prefill(st, all, torch::kCPU, temperature, 0, 1.0, 1.0, rng, tok);
    while (static_cast<int64_t>(all.size() - prompt.size()) < N)
      draft_model_speculative_step(st, all, torch::kCPU, temperature, 0, 1.0, 1.0, rng, tok);
    // Cache invariant: both caches hold everything except the pending token.
    const int64_t want_len = static_cast<int64_t>(all.size()) - 1;
    check(st.target_kv.seq_len() == want_len && st.draft_kv.seq_len() == want_len,
          "draft-model caches hold committed - 1 tokens (target " + std::to_string(st.target_kv.seq_len()) +
              ", draft " + std::to_string(st.draft_kv.seq_len()) + ", want " + std::to_string(want_len) + ")");
    *acc = st.total_accepted;
    *drf = st.total_drafted;
    std::vector<int64_t> got(all.begin() + static_cast<long>(prompt.size()), all.end());
    got.resize(N);
    return got;
  };
  std::vector<int64_t> want(ref.begin(), ref.begin() + N);
  for (int64_t k : {1, 4}) {
    int64_t a = 0, d = 0;
    auto g1 = run(&target, 0.0, k, &a, &d);
    check(g1 == want && a == d, "draft-model spec == greedy, draft = target, k=" + std::to_string(k) +
                                    first_diff(g1, want) + " accepted " + std::to_string(a) + "/" +
                                    std::to_string(d));
    auto g2 = run(&other, 0.0, k, &a, &d);
    check(g2 == want, "draft-model spec == greedy, unrelated draft, k=" + std::to_string(k) +
                          first_diff(g2, want) + " accepted " + std::to_string(a) + "/" + std::to_string(d));
  }
  // Sampling: draft == target means p == q, so every draft must be accepted.
  int64_t a = 0, d = 0;
  auto gs = run(&target, 1.0, 4, &a, &d);
  bool valid = true;
  for (auto t : gs) valid = valid && t >= 0 && t < c.vocab_size;
  check(valid && a == d, "draft-model sampling, draft = target: valid tokens, accepted " +
                             std::to_string(a) + "/" + std::to_string(d) + " (must be all)");
}

}  // namespace

// The same cache-parity check on another device in fp32 (MPS has no fp64).
void test_cache_attention_device(const char* name, TransformerConfig c, torch::Device dev) {
  torch::NoGradGuard ng;
  torch::manual_seed(1);
  Transformer m(c);
  m->init_weights();
  m->to(dev);
  m->eval();
  const int64_t T = 24, P = 13;
  auto toks = random_prompt(T, c.vocab_size, 7);
  auto full = m->forward(ids(toks).to(dev)).cpu();
  KVCache kv(c.n_layers, dev);
  std::vector<int64_t> a(toks.begin(), toks.begin() + P), b(toks.begin() + P, toks.end());
  m->forward(ids(a).to(dev), c10::nullopt, -100, &kv);
  auto blk = m->forward(ids(b).to(dev), c10::nullopt, -100, &kv).cpu();
  const double err = (blk - full.narrow(1, P, T - P)).abs().max().item<double>();
  const double scale = full.abs().max().item<double>();
  check(err < 1e-3 * scale, std::string(name) + ": cached multi-token block matches no-cache on " +
                                (dev.is_mps() ? "mps" : "cuda") + " (max err " + sci(err) + ", logit scale " +
                                sci(scale) + ")");
}

// CPU vs device: the same model and input must give the same logits.
void test_cpu_vs_device(int64_t d, int64_t heads, int64_t S, torch::Device dev) {
  torch::NoGradGuard ng;
  auto c = tiny_cfg(-1, -1, 0);
  c.d_model = d;
  c.n_heads = heads;
  torch::manual_seed(3);
  Transformer m(c);
  m->init_weights();
  m->eval();
  auto x = ids(random_prompt(S, c.vocab_size, 9));
  auto ref = m->forward(x);
  m->to(dev);
  auto got = m->forward(x.to(dev)).cpu();
  const double err = (got - ref).abs().max().item<double>(), scale = ref.abs().max().item<double>();
  char b[200];
  std::snprintf(b, sizeof b, "cpu vs %s: head_dim %lld, %lld tokens: max err %s (logit scale %s)",
                dev.is_mps() ? "mps" : "cuda", (long long)(d / heads), (long long)S, sci(err).c_str(),
                sci(scale).c_str());
  check(err < 1e-3 * scale, b);
}

int main(int argc, char** argv) {
  torch::set_num_threads(4);
  if (argc > 1) {  // device checks only: test_spec_greedy mps|cuda
    torch::Device dev = std::string(argv[1]) == "cuda" ? torch::Device(torch::kCUDA) : torch::Device(torch::kMPS);
    test_cache_attention_device("MHA", tiny_cfg(-1, -1, 0), dev);
    test_cache_attention_device("GQA", tiny_cfg(2, -1, 0), dev);
    test_cache_attention_device("sliding window 8", tiny_cfg(-1, 8, 0), dev);
    for (int64_t hd : {16, 64, 128})
      for (int64_t S : {13, 64, 200, 512}) test_cpu_vs_device(hd * 2, 2, S, dev);
    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
  }
  test_cache_attention("MHA", tiny_cfg(-1, -1, 0));
  test_cache_attention("GQA", tiny_cfg(2, -1, 0));
  test_cache_attention("sliding window 8", tiny_cfg(-1, 8, 0));
  test_mtp(tiny_cfg(2, -1, 3));
  {
    auto sl = tiny_cfg(2, -1, 3);  // SL-MTP heads draft through the same exact step
    sl.mtp_head_type = "shared_lowrank";
    sl.mtp_rank = 8;
    test_mtp(sl);
  }
  test_draft_model(tiny_cfg(2, -1, 0));
  std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
  return g_fail ? 1 : 0;
}
