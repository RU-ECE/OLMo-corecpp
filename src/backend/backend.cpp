#include "olmo_cpp/backend/backend.hpp"

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// Default implementations (pure ATen, matches existing code exactly)
// ---------------------------------------------------------------------------

torch::Tensor IBackend::rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
  auto variance = x.pow(2).mean(-1, true).add(eps);
  auto x_norm = x * torch::rsqrt(variance);
  if (weight.defined()) {
    x_norm = x_norm * weight.to(x.dtype());
  }
  return x_norm;
}

torch::Tensor IBackend::silu_mul(torch::Tensor gate, torch::Tensor up) {
  return torch::silu(gate) * up;
}

torch::Tensor IBackend::apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos) {
  auto chunks = t.chunk(2, -1);
  auto rotated = torch::cat({-chunks[1], chunks[0]}, -1);
  return (t * cos + rotated * sin).to(t.dtype());
}

torch::Tensor IBackend::residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                           torch::Tensor weight, double eps) {
  auto h = x + residual;
  return rms_norm(h, weight, eps);
}

// ---------------------------------------------------------------------------
// Default LibTorch backend (just inherits default impls)
// ---------------------------------------------------------------------------

class LibTorchBackend : public IBackend {
 public:
  const char* name() const override { return "libtorch"; }
};

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static std::unique_ptr<IBackend> g_backend;

IBackend& get_backend() {
  if (!g_backend) {
    g_backend = std::make_unique<LibTorchBackend>();
  }
  return *g_backend;
}

void set_backend(std::unique_ptr<IBackend> backend) {
  g_backend = std::move(backend);
}

}  // namespace olmo_cpp
