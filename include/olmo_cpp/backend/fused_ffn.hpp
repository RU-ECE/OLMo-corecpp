#pragma once

/**
 * include/olmo_cpp/backend/fused_ffn.hpp
 *
 * Fused SwiGLU FFN — item I.
 *
 *   y = w_down * (silu(gate) * up)        where gate, up = split(w_gate_up * x)
 *
 * Today's path: cuBLAS GEMM(x, w_gate_up) -> [B,S,2H] in HBM -> silu_mul
 * kernel reads/writes [B,S,2H]+[B,S,H] -> cuBLAS GEMM(act, w_down)
 * reads [B,S,H] writes [B,S,d]. Three large HBM roundtrips.
 *
 * Fused: keep the [2H]→[H]→[d] chain per row in shared memory. The
 * actual matmuls stay tensor-core-driven in the long-term variant
 * (mma / wgmma). Current commit ships the CPU reference, the host
 * dispatch wiring, and a naive CUDA kernel (one block per row, FMA
 * loops, no tensor cores) — correct, identical numerics, used as the
 * baseline against which the tensor-core variant lands.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// Fused FFN forward. Inputs:
///   x          : [B, S, d]
///   w_gate_up  : [2H, d]   (concatenated gate || up rows along the output)
///   w_down     : [d, H]
/// Output:
///   y          : [B, S, d]
torch::Tensor fused_ffn(torch::Tensor x,
                         torch::Tensor w_gate_up,
                         torch::Tensor w_down);

#ifdef OLMO_HAS_CUDA_KERNELS
torch::Tensor fused_ffn_cuda(torch::Tensor x,
                              torch::Tensor w_gate_up,
                              torch::Tensor w_down);
#endif

torch::Tensor fused_ffn_cpu(torch::Tensor x,
                             torch::Tensor w_gate_up,
                             torch::Tensor w_down);

/// Autograd-aware variant for training call sites.
torch::Tensor fused_ffn_autograd(torch::Tensor x,
                                   torch::Tensor w_gate_up,
                                   torch::Tensor w_down);

}  // namespace olmo_cpp
