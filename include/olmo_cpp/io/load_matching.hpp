#pragma once
/**
 * load_matching — copy a saved module archive into an existing module by name.
 *
 * Walks the module tree alongside the archive (torch::save layout) and copies each
 * tensor the archive has into the module's existing parameter or buffer with copy_,
 * so dtype and device follow the module, not the file: a bf16 archive loads into an
 * fp32 or GPU model. Only the archive is held on the host; no second model is built.
 * (torch::load instead re-points each parameter at the archive's storage, and fails
 * when the dtypes differ.)
 */
#include <torch/torch.h>
#include <string>
#include <vector>

namespace olmo_cpp {

struct LoadMatchingResult {
  int64_t loaded = 0;                 // parameters copied
  std::vector<std::string> missing;   // module parameters the archive did not have
  std::vector<std::string> unexpected;// archive entries the module has no slot for
};

/// Throws on a shape mismatch. Missing parameters are returned, not thrown: the
/// caller decides which are allowed (adapters, fresh MTP heads).
LoadMatchingResult load_matching(torch::nn::Module& module, const std::string& path);

}  // namespace olmo_cpp
