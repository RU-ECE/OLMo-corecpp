/**
 * tools/slmtp_killtest.cpp — does a low-rank SL-MTP head predict as well as a
 * full-rank head? (the speed plan's kill test, frozen-trunk version)
 *
 * The trunk and its LM head are frozen; only the extra heads train, so the
 * trunk runs once:
 *
 *   extract  tokenizes a text file (stories split on <|endoftext|>), runs the
 *            frozen model over fixed-length sequences, and caches per position
 *            the backbone output h (bf16), the token, and the main head's
 *            greedy prediction.
 *   train    trains K heads (head k predicts token t+k+2 from h_t) on the
 *            first (1 - eval_frac) of the sequences and reports, on the rest,
 *            per head: top-1 accuracy against the real token, agreement with
 *            the model's own greedy choice for that token (what greedy
 *            speculative decoding needs to accept a draft), and CE.
 *            --variant full   proj (d x d, init = identity) + RMSNorm, then the
 *                             frozen LM head: what llm-cpp builds today
 *            --variant sl     SL-MTP, rank --rank, learnable a (init 1)
 *            --fix-a0         SL-MTP with a pinned at 0: a pure rank-r head
 *
 *   slmtp_killtest extract --config c.json --ckpt m.pt --vocab v.json --merges m.txt
 *                          --text f.txt --out DIR [--seq 512] [--tokens 600000] [--device mps]
 *   slmtp_killtest train --config c.json --ckpt m.pt --features DIR --variant full|sl
 *                        [--rank 64] [--heads 3] [--epochs 1] [--batch 2] [--lr 1e-3]
 *                        [--eval-frac 0.1] [--fix-a0] [--target real|greedy] [--save heads.pt] [--device mps]
 * Prints one JSON line with the results.
 */
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include "olmo_cpp/model/sl_mtp_head.hpp"
#include "olmo_cpp/model/transformer.hpp"

#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace olmo_cpp;
namespace fs = std::filesystem;

namespace {

std::map<std::string, std::string> parse(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 2; i < argc; ++i) {
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

torch::Device device_of(const std::string& s) {
  return s == "cuda" ? torch::Device(torch::kCUDA) : s == "cpu" ? torch::Device(torch::kCPU) : torch::Device(torch::kMPS);
}

void sync(const torch::Device& d) {
  if (d.is_mps()) torch::mps::synchronize();
  if (d.is_cuda()) torch::cuda::synchronize();
}

Transformer load_model(const std::string& cfg_path, const std::string& ckpt) {
  auto cfg = load_config_from_json(cfg_path);
  cfg.validate();
  Transformer m(cfg);
  torch::load(m, ckpt, torch::kCPU);
  m->eval();
  return m;
}

template <typename T>
void write_raw(const fs::path& p, const torch::Tensor& t) {
  auto c = t.contiguous().cpu();
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(c.data_ptr()), static_cast<std::streamsize>(c.numel() * sizeof(T)));
}

torch::Tensor read_raw(const fs::path& p, torch::ScalarType dt, std::vector<int64_t> shape) {
  auto t = torch::empty(shape, torch::TensorOptions().dtype(dt));
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char*>(t.data_ptr()), static_cast<std::streamsize>(t.numel() * t.element_size()));
  TORCH_CHECK(f.gcount() == static_cast<std::streamsize>(t.numel() * t.element_size()), "short read: ", p.string());
  return t;
}

// ── extract ──────────────────────────────────────────────────────────────
int extract(const std::map<std::string, std::string>& a) {
  const int64_t S = std::stoll(get(a, "seq", "512"));
  const int64_t want = std::stoll(get(a, "tokens", "600000"));
  const auto dev = device_of(get(a, "device", "mps"));
  const fs::path out = get(a, "out");
  fs::create_directories(out);

  BPETokenizer tok;
  TORCH_CHECK(tok.load(get(a, "vocab"), get(a, "merges")), "tokenizer load failed");
  std::ifstream tf(get(a, "text"));
  std::stringstream ss;
  ss << tf.rdbuf();
  const std::string text = ss.str();
  const std::string sep = "<|endoftext|>";
  std::vector<int64_t> ids;
  size_t pos = 0;
  while (pos < text.size() && static_cast<int64_t>(ids.size()) < want + S) {
    size_t e = text.find(sep, pos);
    if (e == std::string::npos) e = text.size();
    std::string story = text.substr(pos, e - pos);
    while (!story.empty() && std::isspace(static_cast<unsigned char>(story.front()))) story.erase(story.begin());
    if (!story.empty()) {
      for (auto id : tok.encode(story)) ids.push_back(static_cast<int64_t>(id));  // encode() ends with EOS
    }
    pos = e + sep.size();
  }
  const int64_t n_seq = std::min<int64_t>(static_cast<int64_t>(ids.size()), want) / S;
  TORCH_CHECK(n_seq > 0, "not enough text");
  auto tokens = torch::tensor(ids, torch::kInt64).narrow(0, 0, n_seq * S).view({n_seq, S});

  auto model = load_model(get(a, "config"), get(a, "ckpt"));
  model->to(dev);
  const int64_t d = model->forward_backbone(tokens.narrow(0, 0, 1).to(dev)).size(-1);
  auto H = torch::empty({n_seq, S, d}, torch::kBFloat16);
  auto greedy = torch::empty({n_seq, S}, torch::kInt32);
  double nll = 0.0;
  int64_t n_pred = 0, n_right = 0;
  torch::NoGradGuard ng;
  auto t0 = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < n_seq; ++i) {
    auto x = tokens.narrow(0, i, 1).to(dev);
    auto h = model->forward_backbone(x);         // [1, S, d]
    auto logits = model->apply_lm_head(h);        // [1, S, V]
    auto lab = tokens.narrow(0, i, 1).narrow(1, 1, S - 1).to(dev).reshape(-1);
    auto lg = logits.narrow(1, 0, S - 1).reshape({S - 1, -1});
    nll += torch::nn::functional::cross_entropy(lg, lab, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum)).item<double>();
    auto am = logits.argmax(-1);                  // [1, S]
    n_right += (am.narrow(1, 0, S - 1).reshape(-1) == lab).sum().item<int64_t>();
    n_pred += S - 1;
    H[i].copy_(h[0].to(torch::kCPU, torch::kBFloat16));
    greedy[i].copy_(am[0].to(torch::kCPU, torch::kInt32));
    if ((i + 1) % 50 == 0 || i + 1 == n_seq) {
      sync(dev);
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr, "extract %lld/%lld seqs, %.0f tok/s, main CE %.3f, top-1 %.3f\n", (long long)(i + 1),
                   (long long)n_seq, (i + 1) * S / s, nll / n_pred, double(n_right) / n_pred);
    }
  }
  write_raw<at::BFloat16>(out / "h.bf16", H);
  write_raw<int64_t>(out / "tokens.i64", tokens);
  write_raw<int32_t>(out / "greedy.i32", greedy);
  std::ofstream(out / "meta.json") << "{\"n_seq\": " << n_seq << ", \"seq\": " << S << ", \"d\": " << d
                                   << ", \"main_ce\": " << nll / n_pred << ", \"main_top1\": "
                                   << double(n_right) / n_pred << "}\n";
  std::printf("{\"n_seq\": %lld, \"seq\": %lld, \"d\": %lld, \"main_ce\": %.4f, \"main_top1\": %.4f}\n",
              (long long)n_seq, (long long)S, (long long)d, nll / n_pred, double(n_right) / n_pred);
  return 0;
}

// ── train ────────────────────────────────────────────────────────────────
struct FullHead {
  torch::Tensor P, gk;  // [d, d] (init identity), [d]
};

int train(const std::map<std::string, std::string>& a) {
  const auto dev = device_of(get(a, "device", "mps"));
  const fs::path dir = get(a, "features");
  const std::string variant = get(a, "variant");
  const int64_t r = std::stoll(get(a, "rank", "64"));
  const int64_t K = std::stoll(get(a, "heads", "3"));
  const int64_t epochs = std::stoll(get(a, "epochs", "1"));
  const int64_t B = std::stoll(get(a, "batch", "2"));
  const double lr = std::stod(get(a, "lr", "1e-3"));
  const double eval_frac = std::stod(get(a, "eval-frac", "0.1"));
  const bool fix_a0 = a.count("fix-a0") > 0;
  const int64_t max_steps = std::stoll(get(a, "max-steps", "0"));
  // --target greedy: self-distillation. Head k learns the model's own greedy pick for token
  // t+k+2 (made at position t+k+1) instead of the real token: greedy speculative decoding
  // accepts a draft exactly when it equals that pick.
  const bool greedy_target = get(a, "target", "real") == "greedy";

  std::ifstream mf(dir / "meta.json");
  std::stringstream ms;
  ms << mf.rdbuf();
  auto meta = nlohmann::json::parse(ms.str());
  const int64_t n_seq = meta["n_seq"], S = meta["seq"], d = meta["d"];
  auto H = read_raw(dir / "h.bf16", torch::kBFloat16, {n_seq, S, d});
  auto tokens = read_raw(dir / "tokens.i64", torch::kInt64, {n_seq, S});
  auto greedy = read_raw(dir / "greedy.i32", torch::kInt32, {n_seq, S}).to(torch::kInt64);

  torch::Tensor W, g;
  double eps;
  {
    auto model = load_model(get(a, "config"), get(a, "ckpt"));
    auto p = model->named_parameters();
    W = p["lm_head.w_out.weight"].detach().to(dev).clone();
    g = p["lm_head.norm.weight"].detach().to(dev).clone();
    eps = load_config_from_json(get(a, "config")).layer_norm_eps;
  }
  const int64_t n_eval = std::max<int64_t>(1, static_cast<int64_t>(n_seq * eval_frac));
  const int64_t n_train = n_seq - n_eval;

  torch::manual_seed(0);
  std::vector<FullHead> full;
  std::vector<SLMTPHead> sl;
  std::vector<torch::Tensor> params;
  for (int64_t k = 0; k < K; ++k) {
    if (variant == "full") {
      FullHead f{torch::eye(d, torch::TensorOptions().device(dev)).requires_grad_(true),
                 torch::ones({d}, torch::TensorOptions().device(dev)).requires_grad_(true)};
      params.push_back(f.P);
      params.push_back(f.gk);
      full.push_back(f);
    } else {
      SLMTPHead h(d, r);
      h->to(dev);
      if (fix_a0) {
        torch::NoGradGuard ng;
        h->a.fill_(0.0);
        h->a.set_requires_grad(false);
        h->U.normal_(0.0, 0.02);  // a pure rank-r head needs a live U from the start
      }
      for (auto& p : h->parameters())
        if (p.requires_grad()) params.push_back(p);
      sl.push_back(h);
    }
  }
  int64_t n_params = 0;
  for (auto& p : params) n_params += p.numel();
  torch::optim::AdamW opt(params, torch::optim::AdamWOptions(lr).weight_decay(0.0));

  auto rmsn = [&](const torch::Tensor& x, const torch::Tensor& w) {
    return x * torch::rsqrt(x.pow(2).mean(-1, true) + eps) * w;
  };
  // Head logits for positions [0, S - shift) of a [b, S, d] batch.
  auto head_logits = [&](int64_t k, const torch::Tensor& h, const torch::Tensor& y) {
    const int64_t shift = k + 1, n = h.size(1) - shift;
    auto hk = h.narrow(1, 0, n);
    if (variant == "full") {
      auto u = rmsn(torch::matmul(hk, full[static_cast<size_t>(k)].P.t()), full[static_cast<size_t>(k)].gk);
      return torch::matmul(rmsn(u, g), W.t());
    }
    const auto& hd = sl[static_cast<size_t>(k)];
    auto c = torch::matmul(hk, hd->V);
    auto u = hd->a * hk + torch::matmul(c, hd->U.t());
    auto T = torch::matmul(W, g.unsqueeze(1) * hd->U);
    return (hd->a * y.narrow(1, 0, n) + torch::matmul(c, T.t())) * torch::rsqrt(u.pow(2).mean(-1, true) + eps);
  };

  auto evaluate = [&]() {
    torch::NoGradGuard ng;
    std::vector<double> acc(K, 0), agree(K, 0), ce(K, 0), n(K, 0);
    for (int64_t i = n_train; i < n_seq; i += B) {
      const int64_t b = std::min(B, n_seq - i);
      auto h = H.narrow(0, i, b).to(dev, torch::kFloat32);
      auto y = variant == "full" ? torch::Tensor() : torch::matmul(h * g, W.t());
      for (int64_t k = 0; k < K; ++k) {
        const int64_t shift = k + 1, m = S - shift - 1;  // targets t+k+2 must exist
        auto lg = head_logits(k, h, y).narrow(1, 0, m);
        auto pred = lg.argmax(-1).cpu();
        auto truth = tokens.narrow(0, i, b).narrow(1, shift + 1, m);
        // The model's own greedy pick for token t+k+2 is made at position t+k+1.
        auto mine = greedy.narrow(0, i, b).narrow(1, shift, m);
        acc[k] += (pred == truth).sum().item<double>();
        agree[k] += (pred == mine).sum().item<double>();
        ce[k] += torch::nn::functional::cross_entropy(
                     lg.reshape({-1, lg.size(-1)}), truth.to(dev).reshape(-1),
                     torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum))
                     .item<double>();
        n[k] += static_cast<double>(b * m);
      }
    }
    nlohmann::json j = nlohmann::json::array();
    for (int64_t k = 0; k < K; ++k)
      j.push_back({{"head", k}, {"predicts", "t+" + std::to_string(k + 2)}, {"top1", acc[k] / n[k]},
                   {"agree_greedy", agree[k] / n[k]}, {"ce", ce[k] / n[k]}});
    return j;
  };

  auto before = evaluate();
  const int64_t steps_per_epoch = (n_train + B - 1) / B;
  int64_t total = epochs * steps_per_epoch;
  if (max_steps > 0) total = std::min(total, max_steps);
  auto t0 = std::chrono::steady_clock::now();
  double last = 0.0;
  for (int64_t step = 0; step < total; ++step) {
    const double frac = static_cast<double>(step) / static_cast<double>(total);
    for (auto& grp : opt.param_groups())
      static_cast<torch::optim::AdamWOptions&>(grp.options()).lr(lr * 0.5 * (1 + std::cos(M_PI * frac)));
    const int64_t i = (step % steps_per_epoch) * B;
    const int64_t b = std::min(B, n_train - i);
    auto h = H.narrow(0, i, b).to(dev, torch::kFloat32);
    auto y = variant == "full" ? torch::Tensor() : torch::matmul(h * g, W.t());  // frozen: no grad
    torch::Tensor loss;
    for (int64_t k = 0; k < K; ++k) {
      const int64_t shift = k + 1, m = S - shift - 1;
      auto lg = head_logits(k, h, y).narrow(1, 0, m);
      auto truth = (greedy_target ? greedy.narrow(0, i, b).narrow(1, shift, m)
                                  : tokens.narrow(0, i, b).narrow(1, shift + 1, m)).to(dev);
      auto l = torch::nn::functional::cross_entropy(lg.reshape({-1, lg.size(-1)}), truth.reshape(-1));
      loss = loss.defined() ? loss + l : l;
    }
    loss = loss / static_cast<double>(K);
    opt.zero_grad();
    loss.backward();
    opt.step();
    if ((step + 1) % 50 == 0 || step + 1 == total) {
      last = loss.item<double>();
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr, "%s r=%lld step %lld/%lld loss %.4f (%.2f s/step)\n", variant.c_str(), (long long)r,
                   (long long)(step + 1), (long long)total, last, s / (step + 1));
    }
  }
  sync(dev);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  auto after = evaluate();
  nlohmann::json out = {{"variant", variant + (fix_a0 ? "-a0" : "")},
                        {"target", greedy_target ? "greedy" : "real"},
                        {"rank", variant == "full" ? d : r},
                        {"trainable_params", n_params},
                        {"lr", lr},
                        {"steps", total},
                        {"train_tokens", total * B * S},
                        {"train_seconds", secs},
                        {"final_train_loss", last},
                        {"before", before},
                        {"after", after}};
  if (variant == "sl") {
    std::vector<double> as;
    for (auto& hd : sl) as.push_back(hd->a.item<double>());
    out["a"] = as;
    if (a.count("save")) {  // U, V, a per head, in head order (tools/bench_spec_decode loads them)
      std::vector<torch::Tensor> ts;
      for (auto& hd : sl)
        for (auto* t : {&hd->U, &hd->V, &hd->a}) ts.push_back(t->detach().cpu());
      torch::save(ts, get(a, "save"));
      out["saved"] = get(a, "save");
    }
  }
  std::printf("%s\n", out.dump().c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: slmtp_killtest extract|train ...  (see the header comment)\n");
    return 2;
  }
  const std::string mode = argv[1];
  auto a = parse(argc, argv);
  if (mode == "extract") return extract(a);
  if (mode == "train") return train(a);
  std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
