#include "olmo_cpp/optim/muon.hpp"

namespace olmo_cpp {

namespace {

/// Per-parameter state for Muon optimizer
struct MuonParamState : public torch::optim::OptimizerCloneableParamState<MuonParamState> {
  TORCH_ARG(torch::Tensor, momentum_buffer);
  TORCH_ARG(int64_t, step) = 0;

  void serialize(torch::serialize::OutputArchive& archive) const override {
    archive.write("step", torch::scalar_tensor(step(), torch::kInt64));
    if (momentum_buffer().defined()) {
      archive.write("momentum_buffer", momentum_buffer());
    }
  }

  void serialize(torch::serialize::InputArchive& archive) override {
    torch::Tensor t;
    archive.read("step", t);
    step(t.item<int64_t>());
    torch::Tensor buf;
    if (archive.try_read("momentum_buffer", buf)) {
      momentum_buffer(buf);
    }
  }
};

}  // namespace

Muon::Muon(std::vector<torch::Tensor> params, MuonOptions defaults)
    : Optimizer(
          {torch::optim::OptimizerParamGroup(std::move(params))},
          std::make_unique<MuonOptions>(defaults)) {}

Muon::Muon(std::vector<torch::optim::OptimizerParamGroup> param_groups, MuonOptions defaults)
    : Optimizer(
          std::move(param_groups),
          std::make_unique<MuonOptions>(defaults)) {}

torch::Tensor Muon::newton_schulz_orthogonalize(torch::Tensor G, int64_t steps) {
  // Newton-Schulz iteration for polar decomposition
  // Finds the nearest orthogonal matrix to G
  //
  // For a tall matrix (rows >= cols), we compute U from the polar decomposition G = U * S
  // For a wide matrix (rows < cols), we transpose, orthogonalize, then transpose back
  bool transposed = false;
  if (G.size(0) < G.size(1)) {
    G = G.t();
    transposed = true;
  }

  // Normalize by Frobenius norm
  auto norm = G.norm();
  if (norm.item<double>() < 1e-12) {
    return transposed ? G.t() : G;
  }
  auto X = G / norm;

  // Newton-Schulz iterations: X = X * (3*I - X^T @ X) / 2
  auto I = torch::eye(X.size(1), X.options());
  for (int64_t i = 0; i < steps; ++i) {
    auto A = torch::mm(X.t(), X);
    X = torch::mm(X, (3.0 * I - A) / 2.0);
  }

  // Scale back by original norm
  X = X * norm;

  if (transposed) {
    X = X.t();
  }
  return X;
}

torch::Tensor Muon::step(LossClosure closure) {
  torch::NoGradGuard no_grad;
  torch::Tensor loss = {};
  if (closure) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  for (auto& group : param_groups_) {
    auto& options = static_cast<MuonOptions&>(group.options());
    const double lr = options.lr();
    const double mom = options.momentum();
    const double weight_decay = options.weight_decay();
    const int64_t ns_steps = options.ns_steps();

    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }

      auto grad = p.grad();
      auto key = p.unsafeGetTensorImpl();

      // Initialize state if needed
      if (state_.find(key) == state_.end()) {
        auto s = std::make_unique<MuonParamState>();
        s->momentum_buffer(torch::zeros_like(p.data()));
        s->step(0);
        state_[key] = std::move(s);
      }

      auto& state = static_cast<MuonParamState&>(*state_[key]);
      auto& buf = state.momentum_buffer();
      state.step(state.step() + 1);

      // Step 1: Weight decay (decoupled)
      if (weight_decay != 0.0) {
        p.data().add_(p.data(), -lr * weight_decay);
      }

      // Step 2: Update momentum buffer: buf = momentum * buf + grad
      buf.mul_(mom).add_(grad);

      // Step 3: Compute update
      torch::Tensor update;
      if (p.dim() == 2) {
        // For 2D (matrix) parameters: apply Newton-Schulz orthogonalization
        update = newton_schulz_orthogonalize(buf, ns_steps);
      } else {
        // For non-matrix parameters: just use the momentum buffer directly
        update = buf;
      }

      // Step 4: Apply update
      p.data().add_(update, -lr);
    }
  }

  return loss;
}

}  // namespace olmo_cpp
