/**
 * tools/test_ddp_accum.cpp — proof test for speed-plan bug 2.
 *
 * Two DDP ranks run in one process (two threads) over a loopback c10d
 * backend. Like NCCL, its allreduce result is only guaranteed after
 * Work::wait(), so the loopback applies the sum at wait time: a gradient
 * that was copied before the wait never sees the reduction.
 *
 * Each rank trains a replica of a tiny model with gradient accumulation
 * exactly as src/train.cpp does: hooks registered once, set_sync_required()
 * true only on the final micro-batch, allreduce_gradients() at end of step,
 * then an SGD update. After every step both replicas must equal a single
 * reference model trained on the union of both ranks' micro-batches with
 * the averaged gradient.
 *
 *   ./build/test_ddp_accum            (A = 1, 2, 4 micro-batches, 3 steps each)
 *   ./build/test_ddp_accum --legacy   (old trainer: never calls set_sync_required;
 *                                      must fail loudly, not train wrongly)
 */
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/distributed/ddp.hpp"
#include "olmo_cpp/model/transformer.hpp"

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/torch.h>

#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace olmo_cpp;

namespace {

// ── Loopback two-rank backend ────────────────────────────────────────────
struct Hub {
  std::mutex mu;
  std::condition_variable cv;
  std::map<int64_t, std::array<at::Tensor, 2>> calls;  // seq -> tensor per rank
  std::set<int64_t> reduced;
  bool aborted = false;  // a rank threw; peers must stop waiting
};

class LoopbackWork : public c10d::Work {
 public:
  LoopbackWork(Hub* hub, int64_t seq) : hub_(hub), seq_(seq) {}
  bool wait(std::chrono::milliseconds /*timeout*/) override {
    std::unique_lock<std::mutex> lk(hub_->mu);
    hub_->cv.wait(lk, [&] {
      auto& c = hub_->calls[seq_];
      return (c[0].defined() && c[1].defined()) || hub_->aborted;
    });
    TORCH_CHECK(hub_->calls[seq_][0].defined() && hub_->calls[seq_][1].defined(),
                "loopback: peer rank aborted");
    if (!hub_->reduced.count(seq_)) {
      auto& c = hub_->calls[seq_];
      TORCH_CHECK(c[0].sizes() == c[1].sizes(), "loopback: ranks issued mismatched collectives at #", seq_);
      auto sum = c[0] + c[1];
      c[0].copy_(sum);
      c[1].copy_(sum);
      hub_->reduced.insert(seq_);
    }
    return true;
  }
  bool isCompleted() override { return false; }

 private:
  Hub* hub_;
  int64_t seq_;
};

class LoopbackBackend : public c10d::Backend {
 public:
  LoopbackBackend(Hub* hub, int rank) : c10d::Backend(rank, 2), hub_(hub) {}
  c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor>& tensors,
                                           const c10d::AllreduceOptions& = c10d::AllreduceOptions()) override {
    TORCH_CHECK(tensors.size() == 1, "loopback: one tensor per allreduce");
    std::lock_guard<std::mutex> lk(hub_->mu);
    const int64_t seq = next_seq_++;
    hub_->calls[seq][static_cast<size_t>(getRank())] = tensors[0];
    hub_->cv.notify_all();
    return c10::make_intrusive<LoopbackWork>(hub_, seq);
  }
  const std::string getBackendName() const override { return "loopback"; }

 private:
  Hub* hub_;
  int64_t next_seq_ = 0;
};

// ── Model and data ───────────────────────────────────────────────────────
TransformerConfig tiny() {
  TransformerConfig c;
  c.d_model = 32;
  c.n_layers = 2;
  c.n_heads = 4;
  c.vocab_size = 97;
  c.hidden_size_multiple_of = 16;
  c.num_mtp_heads = 1;
  return c;
}

Transformer make_model() {
  torch::manual_seed(123);
  Transformer m(tiny());
  m->init_weights();
  m->to(torch::kFloat64);
  m->train();
  return m;
}

// All batches are made up front on one thread: torch's global RNG is not
// safe to seed from both rank threads at once.
using Batch = std::pair<torch::Tensor, torch::Tensor>;
std::map<std::array<int, 3>, Batch> g_batches;

void make_batches(int steps, int A) {
  g_batches.clear();
  for (int step = 0; step < steps; ++step)
    for (int m = 0; m < A; ++m)
      for (int r = 0; r < 2; ++r) {
        torch::manual_seed(1000 + 100 * step + 10 * m + r);
        auto in = torch::randint(0, tiny().vocab_size, {2, 12}, torch::kInt64);
        auto lab = torch::randint(0, tiny().vocab_size, {2, 12}, torch::kInt64);
        g_batches[{r, step, m}] = {in, lab};
      }
}

const Batch& batch(int rank, int step, int micro) { return g_batches.at({rank, step, micro}); }

void sgd(Transformer& m, double lr) {
  torch::NoGradGuard ng;
  for (auto& p : m->parameters())
    if (p.grad().defined()) p.sub_(lr * p.grad());
}

double max_param_diff(Transformer& a, Transformer& b) {
  double d = 0.0;
  auto pa = a->parameters(), pb = b->parameters();
  for (size_t i = 0; i < pa.size(); ++i) d = std::max(d, (pa[i] - pb[i]).abs().max().item<double>());
  return d;
}

// Returns 0 on success; prints the evidence either way.
int run(int A, bool legacy) {
  const int steps = 3;
  const double lr = 0.1;
  Hub hub;
  make_batches(steps, A);
  Transformer ref = make_model();
  Transformer rep[2] = {make_model(), make_model()};
  std::string error[2];

  for (int step = 0; step < steps; ++step) {
    // Reference: mean over ranks of the accumulated gradient.
    for (auto& p : ref->parameters()) p.mutable_grad() = torch::Tensor();
    for (int r = 0; r < 2; ++r)
      for (int m = 0; m < A; ++m) {
        auto [in, lab] = batch(r, step, m);
        (ref->forward(in, lab, -100) / static_cast<double>(A * 2)).backward();
      }
    sgd(ref, lr);
  }

  // Replicas, one thread per rank, all steps.
  std::vector<std::thread> th;
  for (int r = 0; r < 2; ++r) {
    th.emplace_back([&, r] {
      try {
        auto ctx = DDPContext::from_backend(c10::make_intrusive<LoopbackBackend>(&hub, r), r, 2);
        std::vector<torch::Tensor> params;
        for (auto& p : rep[r]->parameters()) params.push_back(p);
        ctx.register_grad_hooks(params);
        for (int step = 0; step < steps; ++step) {
          for (auto& p : params) p.mutable_grad() = torch::Tensor();  // zero_grad(set_to_none)
          for (int m = 0; m < A; ++m) {
            auto [in, lab] = batch(r, step, m);
            if (!legacy) ctx.set_sync_required(m + 1 == A);
            (rep[r]->forward(in, lab, -100) / static_cast<double>(A)).backward();
          }
          ctx.allreduce_gradients(params);
          sgd(rep[r], lr);
        }
      } catch (const std::exception& e) {
        error[r] = e.what();
        std::lock_guard<std::mutex> lk(hub.mu);
        hub.aborted = true;
        hub.cv.notify_all();
      }
    });
  }
  for (auto& t : th) t.join();

  const std::string name = "A=" + std::to_string(A) + (legacy ? " legacy trainer" : "");
  if (!error[0].empty() || !error[1].empty()) {
    const std::string e = !error[0].empty() ? error[0] : error[1];
    const bool loud = e.find("reduced twice") != std::string::npos;
    std::printf("%s  %s: stopped with an error (%s)\n", legacy && loud ? "PASS" : "FAIL", name.c_str(),
                e.substr(0, e.find('\n')).c_str());
    return legacy && loud ? 0 : 1;
  }
  const double d0 = max_param_diff(rep[0], ref), d1 = max_param_diff(rep[1], ref);
  const double d01 = max_param_diff(rep[0], rep[1]);
  const bool ok = d0 < 1e-10 && d1 < 1e-10 && d01 < 1e-10;
  std::printf("%s  %s: after %d steps |rank0-ref| %.1e  |rank1-ref| %.1e  |rank0-rank1| %.1e\n",
              ok && !legacy ? "PASS" : (legacy && A == 1 && ok ? "PASS" : "FAIL"), name.c_str(), steps, d0,
              d1, d01);
  if (legacy) return (A == 1 && ok) ? 0 : 1;  // legacy must either be right (A=1) or fail loudly
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const bool legacy = argc > 1 && std::strcmp(argv[1], "--legacy") == 0;
  int fails = 0;
  for (int A : {1, 2, 4}) fails += run(A, legacy);
  std::printf("\n%s: %d failing case(s)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails ? 1 : 0;
}
