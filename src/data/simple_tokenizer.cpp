#include "olmo_cpp/data/simple_tokenizer.hpp"
#include <fstream>
#include <stdexcept>

namespace olmo_cpp {

bool SimpleTokenizer::load_vocab(const std::string& path) {
  std::ifstream f(path);
  if (!f) return false;
  vocab_.clear();
  next_id_ = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    vocab_[line] = next_id_++;
  }
  return true;
}

bool SimpleTokenizer::save_vocab(const std::string& path) const {
  std::ofstream f(path);
  if (!f) return false;
  std::vector<std::pair<std::string, uint32_t>> sorted(vocab_.begin(),
                                                        vocab_.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
  for (const auto& p : sorted) {
    f << p.first << "\n";
  }
  return true;
}

}  // namespace olmo_cpp
