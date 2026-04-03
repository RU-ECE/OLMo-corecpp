#include "olmo_cpp/model/rope.hpp"
#include <cmath>

namespace olmo_cpp {

RotaryEmbeddingImpl::RotaryEmbeddingImpl(int64_t head_size, int64_t theta)
    : dim_(head_size), theta_(theta) {}

torch::Tensor RotaryEmbeddingImpl::compute_inv_freqs(torch::Device device) {
  auto half_dim = dim_ / 2;
  auto indices = torch::arange(half_dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  indices = indices * 2.0 / static_cast<double>(dim_);
  return 1.0 / torch::pow(static_cast<double>(theta_), indices);
}

torch::Tensor RotaryEmbeddingImpl::rotate_half(torch::Tensor x) {
  auto chunks = x.chunk(2, -1);
  return torch::cat({-chunks[1], chunks[0]}, -1);
}

torch::Tensor RotaryEmbeddingImpl::apply_rotary(
    torch::Tensor t, torch::Tensor sin, torch::Tensor cos) {
  return t * cos + rotate_half(t) * sin;
}

RoPEBuffers RotaryEmbeddingImpl::get_buffers(int64_t seq_len, torch::Device device,
                                              torch::Dtype dtype) {
  // Compute in float32 for precision, then cast once to target dtype
  auto inv_freq = compute_inv_freqs(device);
  auto seq = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto freqs = seq.unsqueeze(1) * inv_freq.unsqueeze(0);
  auto positions = torch::cat({freqs, freqs}, -1);
  RoPEBuffers bufs;
  bufs.pos_sin = positions.sin().to(dtype);
  bufs.pos_cos = positions.cos().to(dtype);
  return bufs;
}

std::pair<torch::Tensor, torch::Tensor> RotaryEmbeddingImpl::apply(
    torch::Tensor q,
    torch::Tensor k,
    const RoPEBuffers& bufs,
    std::optional<int64_t> start_pos) {
  auto q_len = q.size(2);
  auto k_len = k.size(2);
  int64_t q_abs_start = start_pos ? *start_pos : (k_len - q_len);
  int64_t k_abs_start = start_pos ? *start_pos : 0;

  auto sin_q = bufs.pos_sin.slice(0, q_abs_start, q_abs_start + q_len).unsqueeze(0).unsqueeze(0);
  auto cos_q = bufs.pos_cos.slice(0, q_abs_start, q_abs_start + q_len).unsqueeze(0).unsqueeze(0);
  auto sin_k = bufs.pos_sin.slice(0, k_abs_start, k_abs_start + k_len).unsqueeze(0).unsqueeze(0);
  auto cos_k = bufs.pos_cos.slice(0, k_abs_start, k_abs_start + k_len).unsqueeze(0).unsqueeze(0);

  // Buffers are already in target dtype and on target device (set in get_buffers)

  auto q_rot = apply_rotary(q, sin_q, cos_q);
  auto k_rot = apply_rotary(k, sin_k, cos_k);
  return {q_rot, k_rot};
}

}  // namespace olmo_cpp
