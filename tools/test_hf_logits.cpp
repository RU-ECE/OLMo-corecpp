/**
 * tools/test_hf_logits.cpp — llm-cpp logits against Hugging Face logits on the same ids.
 *
 *   test_hf_logits --config c.json --ckpt model.pt --ids ids.npy --ref ref.npy
 *       [--device cpu|cuda|mps] [--bf16 0|1] [--dora 0|1] [--tol 1e-4]
 *
 * ids.npy is int64 [B, S]; ref.npy is float32 [B, S, V] (scripts/hf_logits.py writes
 * both from the HF model). Three paths are checked against the same reference:
 *   full    one forward over all S tokens (the training path)
 *   cached  prefill S-1 tokens into a KVCache, then decode the last one (inference)
 *   dora    the DoRA-wrapped model in train mode: adapters at init must not move logits
 * For each it prints max |diff| / max |ref| and how many positions pick the same
 * greedy token, and exits 1 if any relative error exceeds --tol.
 */
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/io/load_matching.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/transformer.hpp"

#include <cnpy.h>
#include <torch/torch.h>
#include <cstdio>
#include <map>
#include <string>

using namespace olmo_cpp;

namespace {

struct Score { double rel = 0, absd = 0; int64_t agree = 0, total = 0; };

Score compare(const torch::Tensor& got, const torch::Tensor& ref) {
  auto g = got.to(torch::kCPU, torch::kFloat32);
  auto r = ref.to(torch::kFloat32);
  TORCH_CHECK(g.sizes() == r.sizes(), "shape ", g.sizes(), " vs reference ", r.sizes());
  Score s;
  s.absd = (g - r).abs().max().item<double>();
  s.rel = s.absd / r.abs().max().item<double>();
  auto same = g.argmax(-1).eq(r.argmax(-1));
  s.agree = same.sum().item<int64_t>();
  s.total = same.numel();
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i + 1 < argc; i += 2) a[std::string(argv[i]).substr(2)] = argv[i + 1];
  auto get = [&](const std::string& k, const std::string& d = "") {
    auto it = a.find(k);
    if (it != a.end()) return it->second;
    if (d.empty()) { std::fprintf(stderr, "missing --%s\n", k.c_str()); std::exit(2); }
    return d;
  };
  const double tol = std::stod(get("tol", "1e-4"));
  const bool bf16 = get("bf16", "0") == "1";
  const std::string dev_s = get("device", "cpu");
  torch::Device dev = dev_s == "cuda" ? torch::Device(torch::kCUDA)
                    : dev_s == "mps"  ? torch::Device(torch::kMPS)
                                      : torch::Device(torch::kCPU);
  const auto dt = bf16 ? torch::kBFloat16 : torch::kFloat32;

  auto cfg = load_config_from_json(get("config"));
  cfg.num_mtp_heads = 0;
  cfg.validate();
  const bool check_dora = get("dora", "1") == "1";
  // Built in the compute dtype and loaded straight from the file (no fp32 host copy:
  // a 14B model would need 59 GB), then moved to the device.
  auto build = [&](const TransformerConfig& c) {
    const auto prev = torch::get_default_dtype();
    torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(dt));
    Transformer m(c);
    torch::set_default_dtype(prev);
    auto lr = load_matching(*m, get("ckpt"));
    if (!lr.unexpected.empty()) {
      std::fprintf(stderr, "%zu checkpoint tensors have no place in this config, e.g. %s\n",
                   lr.unexpected.size(), lr.unexpected[0].c_str());
      std::exit(1);
    }
    for (auto& k : lr.missing)
      if (k.find("dora_") == std::string::npos) {
        std::fprintf(stderr, "%s is not in the checkpoint\n", k.c_str());
        std::exit(1);
      }
    m->to(dev);
    return m;
  };
  Transformer model = build(cfg);
  model->eval();

  cnpy::NpyArray ia = cnpy::npy_load(get("ids")), ra = cnpy::npy_load(get("ref"));
  TORCH_CHECK(ia.word_size == 8 && ia.shape.size() == 2, "ids.npy must be int64 [B, S]");
  TORCH_CHECK(ra.word_size == 4 && ra.shape.size() == 3, "ref.npy must be float32 [B, S, V]");
  const int64_t B = ia.shape[0], S = ia.shape[1], V = ra.shape[2];
  auto ids = torch::from_blob(ia.data<int64_t>(), {B, S}, torch::kInt64).clone().to(dev);
  auto ref = torch::from_blob(ra.data<float>(), {B, S, V}, torch::kFloat32).clone();
  std::printf("B=%lld S=%lld V=%lld dtype=%s device=%s\n", (long long)B, (long long)S,
              (long long)V, bf16 ? "bf16" : "fp32", dev_s.c_str());

  bool ok = true;
  auto report = [&](const char* name, const Score& s) {
    const bool pass = s.rel <= tol;
    ok = ok && pass;
    std::printf("%-7s max|diff| %.3e  rel %.3e  greedy agree %lld/%lld  %s\n", name, s.absd,
                s.rel, (long long)s.agree, (long long)s.total, pass ? "PASS" : "FAIL");
  };

  torch::NoGradGuard ng;
  report("full", compare(model->forward(ids), ref));

  {
    KVCache cache(cfg.n_layers, dev);
    auto pre = model->forward(ids.narrow(1, 0, S - 1), c10::nullopt, -100, &cache);
    auto last = model->forward(ids.narrow(1, S - 1, 1), c10::nullopt, -100, &cache);
    report("cached", compare(torch::cat({pre, last}, 1), ref));
  }

  if (check_dora) {
    model = nullptr;  // one model on the device at a time
    auto dcfg = cfg;
    dcfg.use_dora = true;
    Transformer dm = build(dcfg);
    dm->train();  // magnitude initialises from the base weights in train mode only
    report("dora", compare(dm->forward(ids), ref));
  }
  return ok ? 0 : 1;
}
