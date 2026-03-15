#pragma once
#include <string>
#include <vector>

namespace olmo_cpp {

/// Configuration for a single data source in a mixture
struct SourceConfig {
  std::string path;           // path to .npy file
  double weight = 1.0;        // relative weight in mixture
  std::string name = "";      // optional name
  int64_t max_tokens = -1;    // max tokens from this source (-1 = all)
};

/// Configuration for a data mixture
struct MixtureConfig {
  std::vector<SourceConfig> sources;
  int64_t seed = 42;

  /// Normalize weights to sum to 1
  std::vector<double> normalized_weights() const;

  /// Load from a text file (one source per line: weight<tab>path)
  static MixtureConfig load_from_file(const std::string& path);
};

}  // namespace olmo_cpp
