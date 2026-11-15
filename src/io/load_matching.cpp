#include "olmo_cpp/io/load_matching.hpp"

#include <unordered_set>

namespace olmo_cpp {
namespace {

void walk(torch::nn::Module& m, torch::serialize::InputArchive& ar, const std::string& prefix,
          LoadMatchingResult& r) {
  // Entries of this archive level the module has no parameter/buffer/child for:
  // e.g. Qwen2 q/k/v biases loaded into a config without attention_bias.
  {
    std::unordered_set<std::string> slots;
    for (auto& p : m.named_parameters(false)) slots.insert(p.key());
    for (auto& b : m.named_buffers(false)) slots.insert(b.key());
    for (auto& c : m.named_children()) slots.insert(c.key());
    for (auto& k : ar.keys()) if (!slots.count(k)) r.unexpected.push_back(prefix + k);
  }
  for (auto& p : m.named_parameters(/*recurse=*/false)) {
    torch::Tensor t;
    if (!ar.try_read(p.key(), t)) { r.missing.push_back(prefix + p.key()); continue; }
    TORCH_CHECK(t.sizes() == p.value().sizes(), "load_matching: shape mismatch for ",
                prefix + p.key(), ": archive ", t.sizes(), " vs module ", p.value().sizes());
    p.value().copy_(t);
    ++r.loaded;
  }
  for (auto& b : m.named_buffers(/*recurse=*/false)) {
    torch::Tensor t;
    if (ar.try_read(b.key(), t, /*is_buffer=*/true) && t.sizes() == b.value().sizes())
      b.value().copy_(t);
  }
  for (auto& c : m.named_children()) {
    torch::serialize::InputArchive sub;
    if (ar.try_read(c.key(), sub)) {
      walk(*c.value(), sub, prefix + c.key() + ".", r);
    } else {
      for (auto& p : c.value()->named_parameters(/*recurse=*/true))
        r.missing.push_back(prefix + c.key() + "." + p.key());
    }
  }
}

}  // namespace

LoadMatchingResult load_matching(torch::nn::Module& module, const std::string& path) {
  torch::serialize::InputArchive ar;
  ar.load_from(path, torch::kCPU);
  torch::NoGradGuard ng;
  LoadMatchingResult r;
  walk(module, ar, "", r);
  return r;
}

}  // namespace olmo_cpp
