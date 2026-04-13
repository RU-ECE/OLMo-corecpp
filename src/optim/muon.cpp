#include "olmo_cpp/optim/muon.hpp"

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace olmo_cpp {

namespace {

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
  bool transposed = false;
  if (G.size(0) < G.size(1)) {
    G = G.t();
    transposed = true;
  }

  auto norm = G.norm();
  if (norm.item<double>() < 1e-12) {
    return transposed ? G.t() : G;
  }
  auto X = G / norm;

  auto I = torch::eye(X.size(1), X.options());
  for (int64_t i = 0; i < steps; ++i) {
    auto A = torch::mm(X.t(), X);
    X = torch::mm(X, (3.0 * I - A) / 2.0);
  }

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
    const bool async_ns = options.async_ns();
    const int64_t async_min = options.async_min_numel();

    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }

      auto grad = p.grad();
      auto key = p.unsafeGetTensorImpl();

      if (state_.find(key) == state_.end()) {
        auto s = std::make_unique<MuonParamState>();
        s->momentum_buffer(torch::zeros_like(p.data()));
        s->step(0);
        state_[key] = std::move(s);
      }

      auto& state = static_cast<MuonParamState&>(*state_[key]);
      auto& buf = state.momentum_buffer();
      state.step(state.step() + 1);

      if (weight_decay != 0.0) {
        p.data().add_(p.data(), -lr * weight_decay);
      }

      buf.mul_(mom).add_(grad);

      torch::Tensor update;

      if (p.dim() == 2) {
#ifdef USE_CUDA
        if (async_ns && p.is_cuda() && p.numel() >= async_min) {
          // ── Async path: apply previous step's ortho, launch current on side stream ──
          auto it = async_states_.find(key);
          if (it != async_states_.end() && it->second.has_pending) {
            // Wait for the side stream to finish previous NS
            it->second.ready_event.block(at::cuda::getCurrentCUDAStream(p.device().index()));
            // Apply the stale-by-1-step orthogonalized update
            p.data().add_(it->second.pending_update, -lr);
          }

          // Launch current NS on side stream
          if (!ns_stream_.has_value()) {
            ns_stream_ = at::cuda::getStreamFromPool(false, p.device().index());
          }

          auto& as = async_states_[key];
          {
            // Record an event on the current stream so the NS stream waits
            // until buf is fully computed before reading it.
            at::cuda::CUDAEvent buf_ready;
            buf_ready.record(at::cuda::getCurrentCUDAStream(p.device().index()));
            buf_ready.block(*ns_stream_);

            c10::cuda::CUDAStreamGuard guard(*ns_stream_);
            as.pending_update = newton_schulz_orthogonalize(buf, ns_steps);
            as.ready_event.record(*ns_stream_);
            as.has_pending = true;
          }
          continue;  // skip synchronous apply — will be applied next step
        }
#endif
        update = newton_schulz_orthogonalize(buf, ns_steps);
      } else {
        update = buf;
      }

      p.data().add_(update, -lr);
    }
  }

  return loss;
}

}  // namespace olmo_cpp
