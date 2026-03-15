#include "olmo_cpp/distributed/context_parallel.hpp"
#include <cmath>
#include <limits>

namespace olmo_cpp {

#ifdef OLMO_HAS_DDP

ContextParallelContext::ContextParallelContext(
    c10::intrusive_ptr<c10d::Backend> backend, int rank, int cp_size)
    : backend_(std::move(backend)), rank_(rank), cp_size_(cp_size) {}

std::optional<ContextParallelContext> ContextParallelContext::create(
    c10::intrusive_ptr<c10d::Backend> backend, int cp_size) {
  if (!backend || cp_size < 2) return std::nullopt;
  return ContextParallelContext(backend, backend->getRank() % cp_size, cp_size);
}

torch::Tensor ContextParallelContext::scatter_sequence(torch::Tensor x) {
  // x: [B, S, D] -> [B, S/cp_size, D]
  auto B = x.size(0);
  auto S = x.size(1);
  auto D = x.size(2);
  int64_t chunk_size = S / cp_size_;
  return x.narrow(1, rank_ * chunk_size, chunk_size).contiguous();
}

torch::Tensor ContextParallelContext::gather_sequence(torch::Tensor x) {
  // x: [B, S/cp, D] -> allgather -> [B, S, D]
  int64_t chunk_len = x.size(1);
  std::vector<at::Tensor> gathered(cp_size_);
  for (int i = 0; i < cp_size_; ++i) {
    gathered[i] = torch::empty_like(x);
  }
  std::vector<std::vector<at::Tensor>> output = {gathered};
  std::vector<at::Tensor> input = {x.contiguous()};
  backend_->allgather(output, input)->wait();

  return torch::cat(gathered, /*dim=*/1);
}

torch::Tensor ContextParallelContext::ring_attention(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  // q,k,v: [B, H, S_local, D]
  auto B = q.size(0);
  auto H = q.size(1);
  auto S_local = q.size(2);
  auto D = q.size(3);

  // Online softmax accumulators
  auto output = torch::zeros_like(q);                    // [B,H,S_local,D]
  auto lse = torch::full({B, H, S_local, 1},            // log-sum-exp tracker
                          -std::numeric_limits<float>::infinity(),
                          q.options());

  // Current KV being processed (starts as local)
  auto cur_k = k.contiguous();
  auto cur_v = v.contiguous();

  int next_rank = (rank_ + 1) % cp_size_;
  int prev_rank = (rank_ - 1 + cp_size_) % cp_size_;

  for (int step = 0; step < cp_size_; ++step) {
    int kv_origin = (rank_ - step + cp_size_) % cp_size_;

    // Compute local attention scores: [B, H, S_local, S_local]
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    auto scores = torch::matmul(q, cur_k.transpose(-2, -1)) * scale;

    // Apply causal mask if needed
    if (causal) {
      // Query positions: [rank_ * S_local, (rank_+1) * S_local)
      // Key positions:   [kv_origin * S_local, (kv_origin+1) * S_local)
      auto q_pos = torch::arange(rank_ * S_local, (rank_ + 1) * S_local, q.device());
      auto k_pos = torch::arange(kv_origin * S_local, (kv_origin + 1) * S_local, q.device());
      // mask: q_pos[i] >= k_pos[j]
      auto mask = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0);  // [S_local, S_local]
      scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0),
                                   -std::numeric_limits<float>::infinity());
    }

    // Online softmax update (numerically stable incremental attention)
    // new_lse = log(exp(old_lse) + exp(block_lse))
    auto block_max = std::get<0>(scores.max(-1, /*keepdim=*/true));  // [B,H,S_local,1]
    auto block_exp = (scores - block_max).exp();                      // [B,H,S_local,S_local]
    auto block_sumexp = block_exp.sum(-1, /*keepdim=*/true);          // [B,H,S_local,1]
    auto block_lse = block_max + block_sumexp.log();                  // [B,H,S_local,1]
    auto block_out = torch::matmul(block_exp, cur_v);                 // [B,H,S_local,D]

    // Combine with running accumulator using log-sum-exp
    auto new_lse = torch::logaddexp(lse, block_lse);
    auto old_weight = (lse - new_lse).exp();
    auto new_weight = (block_lse - new_lse).exp();

    output = output * old_weight + block_out * new_weight;
    lse = new_lse;

    // Ring exchange: send cur_k,cur_v to next, receive from prev
    if (step < cp_size_ - 1) {
      auto recv_k = torch::empty_like(cur_k);
      auto recv_v = torch::empty_like(cur_v);

      // Async send/recv
      std::vector<at::Tensor> send_k_vec = {cur_k};
      std::vector<at::Tensor> recv_k_vec = {recv_k};
      std::vector<at::Tensor> send_v_vec = {cur_v};
      std::vector<at::Tensor> recv_v_vec = {recv_v};

      auto send_k_work = backend_->send(send_k_vec, next_rank, /*tag=*/step * 4);
      auto recv_k_work = backend_->recv(recv_k_vec, prev_rank, /*tag=*/step * 4);
      auto send_v_work = backend_->send(send_v_vec, next_rank, /*tag=*/step * 4 + 1);
      auto recv_v_work = backend_->recv(recv_v_vec, prev_rank, /*tag=*/step * 4 + 1);

      send_k_work->wait();
      recv_k_work->wait();
      send_v_work->wait();
      recv_v_work->wait();

      cur_k = recv_k;
      cur_v = recv_v;
    }
  }

  // Normalize: output already weighted by softmax via online algorithm
  return output;
}

#else  // !OLMO_HAS_DDP

ContextParallelContext::ContextParallelContext(int rank, int cp_size)
    : rank_(rank), cp_size_(cp_size) {}

torch::Tensor ContextParallelContext::scatter_sequence(torch::Tensor x) { return x; }
torch::Tensor ContextParallelContext::gather_sequence(torch::Tensor x) { return x; }
torch::Tensor ContextParallelContext::ring_attention(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool /*causal*/) {
  float scale = 1.0f / std::sqrt(static_cast<float>(q.size(-1)));
  auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;
  return torch::matmul(torch::softmax(scores, -1), v);
}

#endif

}  // namespace olmo_cpp
