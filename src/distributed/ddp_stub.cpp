// Stub implementation when Gloo is not available (pip LibTorch)
#include "olmo_cpp/distributed/ddp.hpp"
#include <vector>

namespace olmo_cpp {

std::optional<DDPContext> DDPContext::init_from_env() {
  return std::nullopt;
}

void DDPContext::broadcast_parameters(std::vector<torch::Tensor>& /*parameters*/) {}

void DDPContext::allreduce_gradients(const std::vector<torch::Tensor>& /*parameters*/) {}

}  // namespace olmo_cpp
