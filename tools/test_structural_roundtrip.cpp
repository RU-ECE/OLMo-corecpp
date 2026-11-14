/**
 * tools/test_structural_roundtrip.cpp — is the Structural tokenizer lossless?
 *
 * For every file under each corpus directory (text extensions, <= 1 MB):
 * decode(encode(text)) must equal the text (the trailing EOS removed). Prints
 * the first mismatching file and byte offset with context. Exit 0 = lossless.
 *
 *   ./build/test_structural_roundtrip <structural-config-dir> <vocab.json> <merges.txt> <dir>...
 */
#include "olmo_cpp/data/structural_tokenizer.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <structural-config-dir> <vocab.json> <merges.txt> <dir>...\n", argv[0]);
    return 2;
  }
  olmo_cpp::StructuralTokenizer tok;
  if (!tok.load(argv[1], argv[2], argv[3])) {
    std::fprintf(stderr, "failed to load the structural tokenizer\n");
    return 2;
  }
  int files = 0, bad = 0;
  size_t bytes = 0;
  for (int a = 4; a < argc; ++a) {
    for (const auto& e : fs::recursive_directory_iterator(argv[a])) {
      if (!e.is_regular_file() || e.file_size() > (1u << 20)) continue;
      const auto ext = e.path().extension().string();
      if (ext != ".cpp" && ext != ".hpp" && ext != ".h" && ext != ".cc" && ext != ".py" && ext != ".txt" &&
          ext != ".md" && ext != ".json" && ext != ".js" && ext != ".mjs")
        continue;
      std::ifstream f(e.path(), std::ios::binary);
      std::stringstream ss;
      ss << f.rdbuf();
      const std::string text = ss.str();
      auto ids = tok.encode(text);
      if (!ids.empty() && ids.back() == tok.eos_id()) ids.pop_back();
      const std::string back = tok.decode(ids);
      ++files;
      bytes += text.size();
      if (back != text) {
        size_t i = 0;
        while (i < text.size() && i < back.size() && text[i] == back[i]) ++i;
        if (bad < 3) {
          auto ctx = [&](const std::string& s) {
            std::string c = s.substr(i > 20 ? i - 20 : 0, 50);
            for (auto& ch : c) if (ch == '\n') ch = '|';
            return c;
          };
          std::printf("MISMATCH %s at byte %zu\n  want: %s\n  got:  %s\n", e.path().c_str(), i,
                      ctx(text).c_str(), ctx(back).c_str());
        }
        ++bad;
      }
    }
  }
  std::printf("%s: %d/%d files round-trip exactly (%zu bytes)\n", bad ? "LOSSY" : "LOSSLESS", files - bad, files,
              bytes);
  return bad ? 1 : 0;
}
