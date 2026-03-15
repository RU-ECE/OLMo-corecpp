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
