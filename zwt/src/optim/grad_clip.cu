// Multi-tensor FP32 sum-of-squares reduction and in-place scale.
//
// Used by global gradient clipping. Walks a device-side array of pointers +
// sizes, chunk-strides across them, reduces inside each block, atomicAdds
// to a single fp32 scalar. Scale is a second pass with the same launch shape.

#include <cuda_runtime.h>

namespace zwt::optim::k {

namespace {

constexpr int kBlock = 256;

__device__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

__device__ float block_sum(float v) {
  __shared__ float sh[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : 0.f;
  if (wid == 0) v = warp_sum(v);
  return v;
}

__global__ void k_sumsq(float** ptrs, const int64_t* sizes, int n_tensors,
                        float* out) {
  // One block owns one tensor. For big tensors, the block strides over the
  // elements. This is wasteful for small tensors but 3B has ~few hundred
  // params, each large enough that this is fine.
  int t = blockIdx.x;
  if (t >= n_tensors) return;
  float* p = ptrs[t];
  int64_t n = sizes[t];
  float acc = 0.f;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    float v = p[i];
    acc += v * v;
  }
  acc = block_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc);
}

__global__ void k_scale(float** ptrs, const int64_t* sizes, int n_tensors,
                        float alpha) {
  int t = blockIdx.x;
  if (t >= n_tensors) return;
  float* p = ptrs[t];
  int64_t n = sizes[t];
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) p[i] *= alpha;
}

}  // namespace

void sumsq_fp32_many(float** ptrs, const int64_t* sizes, int n_tensors,
                     float* out, cudaStream_t s) {
  k_sumsq<<<n_tensors, kBlock, 0, s>>>(ptrs, sizes, n_tensors, out);
}

void scale_fp32_many(float** ptrs, const int64_t* sizes, int n_tensors,
                     float alpha, cudaStream_t s) {
  k_scale<<<n_tensors, kBlock, 0, s>>>(ptrs, sizes, n_tensors, alpha);
}

}  // namespace zwt::optim::k
