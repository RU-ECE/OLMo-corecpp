#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace olmo_cpp {

/// Simple tokenizer: whitespace + punctuation split. Builds vocab from corpus.
/// Use for local text when GPT-2 compatibility is not required.
class SimpleTokenizer {
 public:
  static constexpr uint32_t kPadId = 0;
  static constexpr uint32_t kEosId = 1;
  static constexpr uint32_t kUnkId = 2;
  static constexpr uint32_t kFirstUserId = 3;

  SimpleTokenizer() : next_id_(kFirstUserId) {
    vocab_["<pad>"] = kPadId;
    vocab_["<eos>"] = kEosId;
    vocab_["<unk>"] = kUnkId;
  }

  /// Tokenize text and return token IDs. Adds <eos> at end.
  std::vector<uint32_t> encode(const std::string& text) {
    std::vector<uint32_t> ids;
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::string token;
    for (size_t i = 0; i <= lower.size(); ++i) {
      char c = (i < lower.size()) ? lower[i] : '\0';
      if (std::isspace(static_cast<unsigned char>(c)) || c == '\0' ||
          is_punct(c)) {
        if (!token.empty()) {
          ids.push_back(get_or_add(token));
          token.clear();
        }
        if (c == '\0') break;
      } else {
        token += c;
      }
    }
    ids.push_back(kEosId);
    return ids;
  }

  /// Tokenize text and append to output vector (no trailing EOS).
  void encode_append(const std::string& text, std::vector<uint32_t>& out) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::string token;
    for (size_t i = 0; i <= lower.size(); ++i) {
      char c = (i < lower.size()) ? lower[i] : '\0';
      if (std::isspace(static_cast<unsigned char>(c)) || c == '\0' ||
          is_punct(c)) {
        if (!token.empty()) {
          out.push_back(get_or_add(token));
          token.clear();
        }
        if (c == '\0') break;
      } else {
        token += c;
      }
    }
  }

  uint32_t vocab_size() const { return static_cast<uint32_t>(vocab_.size()); }

  /// Load vocab from file: one token per line, id = line number (0-based).
  bool load_vocab(const std::string& path);

  /// Save vocab to file for inspection.
  bool save_vocab(const std::string& path) const;

 private:
  static bool is_punct(char c) {
    return c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':' ||
           c == '"' || c == '\'' || c == '(' || c == ')' || c == '-' || c == '/';
  }

  uint32_t get_or_add(const std::string& token) {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) return it->second;
    uint32_t id = next_id_++;
    vocab_[token] = id;
    return id;
  }

  std::unordered_map<std::string, uint32_t> vocab_;
  uint32_t next_id_;
};

}  // namespace olmo_cpp
