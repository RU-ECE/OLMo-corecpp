// Fused multi-tensor AdamW.
//
// One kernel handles all parameters of a model in a single launch. Each block
// picks up one "chunk" of work; chunks are strided across all tensors so the
// SM occupancy is high even when per-tensor sizes are wildly uneven (which is
// always the case: embedding >> per-layer weight >> per-layer bias).
//
// Param dtype is bf16; gradient, momentum, variance are fp32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace zwt::optim::k {

namespace {

constexpr int kChunkSize = 4096;  // elements per block

__device__ __forceinline__ float bf_to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f_to_bf(float v) { return __float2bfloat16(v); }

__global__ void adamw_kernel(
    void** p_ptrs, float** g_ptrs, float** m_ptrs, float** v_ptrs,
    const int64_t* sizes, int n_tensors,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2_sqrt,
    int64_t total_chunks,
    const int64_t* chunk_tensor_idx, const int64_t* chunk_offsets) {
  int64_t chunk_id = int64_t(blockIdx.x);
  if (chunk_id >= total_chunks) return;

  int t = static_cast<int>(chunk_tensor_idx[chunk_id]);
  int64_t off = chunk_offsets[chunk_id];
  int64_t size = sizes[t];
  int64_t remain = size - off;
  int64_t work = remain < kChunkSize ? remain : kChunkSize;

  __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(p_ptrs[t]) + off;
  float* g = g_ptrs[t] + off;
  float* m = m_ptrs[t] + off;
  float* v = v_ptrs[t] + off;

  // step_size folds bc2_sqrt / bc1 into lr so we avoid per-element division.
  float step_size = lr * bc2_sqrt / bc1;
  float eps_scaled = eps * bc2_sqrt;

  for (int64_t i = threadIdx.x; i < work; i += blockDim.x) {
    float pv = bf_to_f(p[i]);
    float gv = g[i];
    float mv = beta1 * m[i] + (1.f - beta1) * gv;
    float vv = beta2 * v[i] + (1.f - beta2) * gv * gv;
    m[i] = mv;
    v[i] = vv;
    // AdamW: decoupled decay on the parameter itself.
    pv = pv - lr * wd * pv - step_size * mv / (__fsqrt_rn(vv) + eps_scaled);
    p[i] = f_to_bf(pv);
  }
}

// Pack chunks on the host. This is a tiny one-time cost per step (O(n_tensors)).
}  // namespace

void adamw_multi_tensor_bf16(
    void** p_ptrs, float** g_ptrs, float** m_ptrs, float** v_ptrs,
    const int64_t* sizes, int n_tensors,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2_sqrt,
    cudaStream_t s) {
  // Pull sizes back to host to chunk. This is tiny (n_tensors longs, <1 KiB).
  // We could cache this on the CPU side of the AdamW object, but since
  // sizes don't change we can do it once on first launch and reuse.
  static int cached_n = 0;
  static int64_t* d_chunk_tensor_idx = nullptr;
  static int64_t* d_chunk_offsets = nullptr;
  static int64_t cached_chunks = 0;

  if (cached_n != n_tensors) {
    if (d_chunk_tensor_idx) cudaFree(d_chunk_tensor_idx);
    if (d_chunk_offsets)    cudaFree(d_chunk_offsets);

    // Pull sizes to host.
    int64_t* h_sizes = new int64_t[n_tensors];
    cudaMemcpy(h_sizes, sizes, sizeof(int64_t) * n_tensors, cudaMemcpyDeviceToHost);

    int64_t total_chunks = 0;
    for (int i = 0; i < n_tensors; ++i) {
      int64_t n = h_sizes[i];
      total_chunks += (n + kChunkSize - 1) / kChunkSize;
    }

    int64_t* h_idx  = new int64_t[total_chunks];
    int64_t* h_off  = new int64_t[total_chunks];
    int64_t cur = 0;
    for (int i = 0; i < n_tensors; ++i) {
      int64_t n = h_sizes[i];
      int64_t off = 0;
      while (off < n) {
        h_idx[cur] = i;
        h_off[cur] = off;
        ++cur;
        off += kChunkSize;
      }
    }

    cudaMalloc(&d_chunk_tensor_idx, sizeof(int64_t) * total_chunks);
    cudaMalloc(&d_chunk_offsets,    sizeof(int64_t) * total_chunks);
    cudaMemcpyAsync(d_chunk_tensor_idx, h_idx, sizeof(int64_t) * total_chunks,
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_chunk_offsets,    h_off, sizeof(int64_t) * total_chunks,
                    cudaMemcpyHostToDevice, s);

    delete[] h_sizes; delete[] h_idx; delete[] h_off;
    cached_n = n_tensors;
    cached_chunks = total_chunks;
  }

  dim3 grid(static_cast<unsigned>(cached_chunks));
  dim3 block(256);
  adamw_kernel<<<grid, block, 0, s>>>(
      p_ptrs, g_ptrs, m_ptrs, v_ptrs, sizes, n_tensors,
      lr, beta1, beta2, eps, wd, bc1, bc2_sqrt,
      cached_chunks, d_chunk_tensor_idx, d_chunk_offsets);
}

}  // namespace zwt::optim::k
