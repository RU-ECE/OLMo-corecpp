#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <utility>

namespace olmo_cpp {

/// GPT-2 style BPE tokenizer. Load vocab.json + merges.txt for full compatibility.
/// Includes proper pre-tokenization (GPT-2 regex pattern) so spaces are merged
/// with following words, not left as standalone "B" tokens.
class BPETokenizer {
 public:
  BPETokenizer() = default;

  /// Load from GPT-2 format: vocab.json (token->id) and merges.txt (merge pairs).
  /// Returns false on error.
  bool load(const std::string& vocab_path, const std::string& merges_path);

  /// Encode text to token IDs. Adds EOS (50256 for GPT-2) at end.
  std::vector<uint32_t> encode(const std::string& text);

  /// Encode and append to output (no trailing EOS).
  void encode_append(const std::string& text, std::vector<uint32_t>& out);

  /// Decode token IDs to text.
  std::string decode(const std::vector<uint32_t>& ids);
  std::string decode(const std::vector<int64_t>& ids);

  /// For models trained with the old tokenizer (spaces encoded as "B" id 33).
  /// When true, decode token 33 as space instead of "B".
  void set_legacy_decode(bool v) { legacy_decode_ = v; }
  bool legacy_decode() const { return legacy_decode_; }

  uint32_t vocab_size() const { return static_cast<uint32_t>(vocab_.size()); }
  uint32_t eos_id() const { return eos_id_; }

  /// Expose pre-tokenization chunks for inspection (what BPE sees before merging).
  std::vector<std::string> get_pre_tokenized_chunks(const std::string& text);

  /// Decode a single token ID to its raw string representation.
  std::string decode_token(uint32_t id) const;

 private:
  void init_bytes_to_unicode();

  /// GPT-2 pre-tokenization: split text into chunks that respect word boundaries.
  /// Each chunk is BPE-encoded independently, preventing cross-word merges.
  std::vector<std::string> pre_tokenize(const std::string& text);

  /// Apply BPE merges to a pre-tokenized chunk.
  std::vector<uint32_t> bpe_encode_chunk(const std::string& chunk);

  std::unordered_map<std::string, uint32_t> vocab_;
  std::vector<std::string> id_to_token_;  // for decode
  std::vector<std::pair<std::string, std::string>> merges_;
  std::unordered_map<std::string, int> merge_ranks_;  // "a\0b" -> priority rank (lower = higher priority)
  std::unordered_map<std::string, std::string> bytes_to_unicode_;
  std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;  // reverse mapping for decode
  uint32_t eos_id_ = 50256;
  bool legacy_decode_ = false;  // decode token 33 as space (for old-trained checkpoints)
};

}  // namespace olmo_cpp
