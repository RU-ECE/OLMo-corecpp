#pragma once

#include "zwt/core/tensor.hpp"

namespace zwt::ops {

// All ops assume tensors are contiguous and live on the same device.
// Shapes must match. In-place versions update the first argument.

// y = x + bias (broadcast over the last dim or none if shapes match)
void add_bias(Tensor& y, const Tensor& bias);

// grad_bias += sum over all rows of grad_y.
// grad_y is [rows, cols] bf16; grad_bias is [cols] fp32 (master gradient).
void bias_backward(const Tensor& grad_y, Tensor& grad_bias);

// Transpose between head-major [B,H,S,D] and token-major [B,S,H,D] layouts.
// Allocates into `out` which must already be shaped correctly.
void transpose_bshd_to_bhsd(const Tensor& in, Tensor& out);
void transpose_bhsd_to_bshd(const Tensor& in, Tensor& out);

// y += x * alpha
void axpy(Tensor& y, const Tensor& x, float alpha);

// y = alpha * y
void scale(Tensor& y, float alpha);

// Fused residual add (out = a + b).
void add(Tensor& out, const Tensor& a, const Tensor& b);

// y = dropout_scale * mask * x (mask stored in-place via 1-bit bools)
// Forward returns mask; backward re-applies it.
void dropout(Tensor& y, const Tensor& x, Tensor& mask_u8, float p, uint64_t seed);

// SiLU gate: out = silu(gate) * up. Fused — two reads, one write per element.
void silu_mul(Tensor& out, const Tensor& gate, const Tensor& up);

// Backward: given grad_out, gate, up produce grad_gate, grad_up.
void silu_mul_backward(const Tensor& grad_out, const Tensor& gate, const Tensor& up,
                       Tensor& grad_gate, Tensor& grad_up);

}  // namespace zwt::ops
