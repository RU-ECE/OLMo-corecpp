#pragma once

#include "zwt/layers/parameter.hpp"

#include <vector>

namespace zwt::optim {

// Global L2-norm gradient clipping. Computes norm of fp32 master-grads across
// all parameters, scales in-place if norm > max_norm.
//
// Returns the unclipped global L2 norm (useful for logging).
// max_norm <= 0 is a no-op (norm is still computed and returned).
float clip_grad_norm(const std::vector<Parameter*>& params, float max_norm);

}  // namespace zwt::optim
