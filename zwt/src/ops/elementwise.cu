#include "zwt/src/ops/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace zwt::ops::k {

namespace {

__device__ __forceinline__ float bf_to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f_to_bf(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

__global__ void k_scale_bf16(__nv_bfloat16* y, float a, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = f_to_bf(bf_to_f(y[i]) * a);
}

__global__ void k_axpy_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, float a, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = f_to_bf(bf_to_f(y[i]) + a * bf_to_f(x[i]));
}

__global__ void k_add_bf16(__nv_bfloat16* out, const __nv_bfloat16* a,
                           const __nv_bfloat16* b, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = f_to_bf(bf_to_f(a[i]) + bf_to_f(b[i]));
}

__global__ void k_add_bias_bf16(__nv_bfloat16* y, const __nv_bfloat16* bias,
                                int64_t rows, int64_t cols) {
  int64_t r = blockIdx.x;
  int64_t c = blockIdx.y * blockDim.x + threadIdx.x;
  if (r >= rows || c >= cols) return;
  y[r * cols + c] = f_to_bf(bf_to_f(y[r * cols + c]) + bf_to_f(bias[c]));
}

__global__ void k_silu_mul_bf16(__nv_bfloat16* out, const __nv_bfloat16* gate,
                                const __nv_bfloat16* up, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = bf_to_f(gate[i]);
  float u = bf_to_f(up[i]);
  out[i] = f_to_bf(silu(g) * u);
}

__global__ void k_silu_mul_bwd_bf16(const __nv_bfloat16* grad_out,
                                    const __nv_bfloat16* gate,
                                    const __nv_bfloat16* up,
                                    __nv_bfloat16* grad_gate,
                                    __nv_bfloat16* grad_up, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = bf_to_f(gate[i]);
  float u = bf_to_f(up[i]);
  float go = bf_to_f(grad_out[i]);
  float sig = 1.0f / (1.0f + __expf(-g));
  float s = g * sig;
  float dsilu = sig * (1.0f + g * (1.0f - sig));
  grad_gate[i] = f_to_bf(go * u * dsilu);
  grad_up[i]   = f_to_bf(go * s);
}

// grad_bias[c] += sum_r grad_y[r, c].
// One block per column, threads reduce across rows, write accumulates into
// a fp32 grad_bias (master gradient for a bf16 param).
__global__ void k_bias_backward_bf16(const __nv_bfloat16* grad_y,
                                     float* grad_bias,
                                     int64_t rows, int64_t cols) {
  int64_t c = blockIdx.x;
  if (c >= cols) return;
  float acc = 0.f;
  for (int64_t r = threadIdx.x; r < rows; r += blockDim.x) {
    acc += bf_to_f(grad_y[r * cols + c]);
  }
  // Block reduce into one value. 32-lane warp shuffle then shared.
  __shared__ float sh[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) sh[wid] = acc;
  __syncthreads();
  float v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : 0.f;
  if (wid == 0) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) grad_bias[c] += v;
  }
}

inline dim3 grid_1d(int64_t n, int block = 256) {
  return dim3(static_cast<unsigned>((n + block - 1) / block));
}

}  // namespace

void scale_bf16(__nv_bfloat16* y, float a, int64_t n, cudaStream_t s) {
  k_scale_bf16<<<grid_1d(n), 256, 0, s>>>(y, a, n);
}

void axpy_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, float a, int64_t n, cudaStream_t s) {
  k_axpy_bf16<<<grid_1d(n), 256, 0, s>>>(y, x, a, n);
}

void add_bf16(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b,
              int64_t n, cudaStream_t s) {
  k_add_bf16<<<grid_1d(n), 256, 0, s>>>(out, a, b, n);
}

void add_bias_bf16(__nv_bfloat16* y, const __nv_bfloat16* bias, int64_t rows,
                   int64_t cols, cudaStream_t s) {
  dim3 block(256);
  dim3 grid(static_cast<unsigned>(rows),
            static_cast<unsigned>((cols + 255) / 256));
  k_add_bias_bf16<<<grid, block, 0, s>>>(y, bias, rows, cols);
}

void silu_mul_bf16(__nv_bfloat16* out, const __nv_bfloat16* gate,
                   const __nv_bfloat16* up, int64_t n, cudaStream_t s) {
  k_silu_mul_bf16<<<grid_1d(n), 256, 0, s>>>(out, gate, up, n);
}

void silu_mul_backward_bf16(const __nv_bfloat16* grad_out,
                            const __nv_bfloat16* gate, const __nv_bfloat16* up,
                            __nv_bfloat16* grad_gate, __nv_bfloat16* grad_up,
                            int64_t n, cudaStream_t s) {
  k_silu_mul_bwd_bf16<<<grid_1d(n), 256, 0, s>>>(grad_out, gate, up,
                                                  grad_gate, grad_up, n);
}

void bias_backward_bf16(const __nv_bfloat16* grad_y, float* grad_bias,
                        int64_t rows, int64_t cols, cudaStream_t s) {
  unsigned blocks = static_cast<unsigned>(cols);
  k_bias_backward_bf16<<<blocks, 256, 0, s>>>(grad_y, grad_bias, rows, cols);
}

// BSHD -> BHSD transpose. out[b,h,s,d] = in[b,s,h,d]
// Data is contiguous in both layouts, only the interior two dims swap order.
__global__ void k_transpose_bshd_bhsd(const __nv_bfloat16* in,
                                      __nv_bfloat16* out,
                                      int64_t B, int64_t S, int64_t H, int64_t D) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = B * S * H * D;
  if (tid >= total) return;
  int64_t d = tid % D;
  int64_t h = (tid / D) % H;
  int64_t s = (tid / (D * H)) % S;
  int64_t b = tid / (D * H * S);
  int64_t dst = ((b * H + h) * S + s) * D + d;
  out[dst] = in[tid];
}

__global__ void k_transpose_bhsd_bshd(const __nv_bfloat16* in,
                                      __nv_bfloat16* out,
                                      int64_t B, int64_t S, int64_t H, int64_t D) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = B * H * S * D;
  if (tid >= total) return;
  int64_t d = tid % D;
  int64_t s = (tid / D) % S;
  int64_t h = (tid / (D * S)) % H;
  int64_t b = tid / (D * S * H);
  int64_t dst = ((b * S + s) * H + h) * D + d;
  out[dst] = in[tid];
}

void transpose_bshd_bhsd_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                              int64_t B, int64_t S, int64_t H, int64_t D,
                              cudaStream_t s) {
  int64_t total = B * S * H * D;
  int block = 256;
  unsigned blocks = static_cast<unsigned>((total + block - 1) / block);
  k_transpose_bshd_bhsd<<<blocks, block, 0, s>>>(in, out, B, S, H, D);
}

void transpose_bhsd_bshd_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                              int64_t B, int64_t S, int64_t H, int64_t D,
                              cudaStream_t s) {
  int64_t total = B * H * S * D;
  int block = 256;
  unsigned blocks = static_cast<unsigned>((total + block - 1) / block);
  k_transpose_bhsd_bshd<<<blocks, block, 0, s>>>(in, out, B, S, H, D);
}

}  // namespace zwt::ops::k
