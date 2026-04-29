/**
 * src/distributed/ddp.cpp
 *
 * ─── What DDP is ────────────────────────────────────────────────────
 *
 * DDP = **D**istributed **D**ata **P**arallel — the simplest way to
 * scale training across GPUs.
 *
 * Each GPU (called a "rank") holds an IDENTICAL copy of the model.
 * They process DIFFERENT microbatches of data in parallel. After
 * backward, each rank has its own gradient that reflects only its
 * microbatch. To stay synchronised, an **all_reduce(SUM)** sums the
 * gradients across all ranks, then each rank divides by world_size to
 * get the average gradient. Since the model and optimizer were
 * identical going in, applying the averaged gradient keeps them
 * identical going out.
 *
 *   per step: forward -> backward -> all_reduce(grads) -> optim.step()
 *
 * One allreduce per step, regardless of how many parameters — the
 * communication is overlap-friendly with backward (you can start
 * reducing the gradient of layer N while still backpropping through
 * layer N-1).
 *
 * ─── How init works here ────────────────────────────────────────────
 *
 * DDPContext::init_from_env() reads the standard env vars:
 *   MASTER_ADDR / MASTER_PORT   where rank 0 listens
 *   RANK         my rank in [0, WORLD_SIZE)
 *   WORLD_SIZE   total number of ranks
 * It opens a c10d::TCPStore on rank 0 (the rendezvous service), then
 * constructs a c10d::ProcessGroupGloo. Returns std::nullopt if the
 * env vars are absent so single-process runs are unaffected.
 *
 * Why **Gloo** and not NCCL? Gloo is CPU-friendly and ships with
 * pip-installed LibTorch. NCCL is faster on GPU clusters but adds a
 * dep that not every developer has. This codebase uses Gloo for
 * portability; rebuild against NCCL for production multi-node runs.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/distributed/ddp.hpp : DDPContext declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: DDPContext::init_from_env() at startup; allreduce
 *     is called inside the "allreduce" ProfileScope after backward.
 *
 * --- Role in training pipeline ---
 *   Compiled when OLMO_USE_DDP=ON. Without it ddp_stub.cpp takes its
 *   place. The quickstart's 3060 single-GPU flow uses the stub.
 */
#include "olmo_cpp/distributed/ddp.hpp"
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace olmo_cpp {

std::optional<DDPContext> DDPContext::init_from_env() {
  const char* rank_str = std::getenv("RANK");
  const char* world_size_str = std::getenv("WORLD_SIZE");
  const char* master_addr = std::getenv("MASTER_ADDR");
  const char* master_port = std::getenv("MASTER_PORT");

  if (!rank_str || !world_size_str || !master_addr || !master_port) {
    return std::nullopt;
  }

  int rank = std::stoi(rank_str);
  int world_size = std::stoi(world_size_str);
  uint16_t port = static_cast<uint16_t>(std::stoi(master_port));

  c10d::TCPStoreOptions store_opts;
  store_opts.port = port;
  store_opts.isServer = (rank == 0);
  store_opts.numWorkers = world_size;
  store_opts.waitWorkers = true;

  auto store = c10::make_intrusive<c10d::TCPStore>(std::string(master_addr), store_opts);

  auto options = c10d::ProcessGroupGloo::Options::create();
  auto backend = c10::make_intrusive<c10d::ProcessGroupGloo>(store, rank, world_size, options);

  return DDPContext(backend, rank, world_size);
}

DDPContext::DDPContext(c10::intrusive_ptr<c10d::Backend> backend, int rank, int world_size)
    : backend_(std::move(backend)), rank_(rank), world_size_(world_size) {}

void DDPContext::broadcast_parameters(std::vector<torch::Tensor>& parameters) {
  if (!backend_) return;
  for (auto& p : parameters) {
    if (p.defined()) {
      std::vector<at::Tensor> tensors = {p};
      c10d::BroadcastOptions opts;
      opts.rootRank = 0;
      backend_->broadcast(tensors, opts)->wait();
    }
  }
}

void DDPContext::allreduce_gradients(const std::vector<torch::Tensor>& parameters) {
  if (!backend_) return;

  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  for (const auto& p : parameters) {
    if (p.defined() && p.requires_grad() && p.grad().defined()) {
      std::vector<at::Tensor> grads = {p.grad()};
      works.push_back(backend_->allreduce(grads));
    }
  }

  for (auto& w : works) {
    w->wait();
  }

  for (const auto& p : parameters) {
    if (p.defined() && p.requires_grad() && p.grad().defined()) {
      p.grad().div_(world_size_);
    }
  }
}

}  // namespace olmo_cpp
