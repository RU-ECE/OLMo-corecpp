// kernels/sl_mtp_ce.cu — row kernels for the chunked SL-MTP loss
// (src/model/sl_mtp_head.cpp, SLMTPChunkedCE). One thread block per row.
//
// A head's logits are z = a*y + p, scaled per row by s (= rsqrt(ms(u)+eps)),
// where y = W(g*h) is the block's shared vocabulary product and p = c T^T its
// low-rank part. z is never materialised: both passes re-form it from y and p.
//
//   pass 1 (rowstats):  lse_i = logsumexp_v(s_i z_iv),  picked_i = s_i z_i,label
//   pass 2 (grad):      G_iv  = w_i (exp(s_i z_iv - lse_i) - [v == label_i])
//                       gs_iv = G_iv s_i                (optional output, for dck/dT GEMMs)
//                       dy_iv += coef * gs_iv           (fp32, in place)
//                       gz_i  = sum_v G_iv z_iv         (for d rsqrt(ms(u)))
//                       gsy_i = sum_v gs_iv y_iv        (for d a)
// The main head is the same call with a = 0, p = y, coef = 1.
// w_i is 0 on ignored rows. All accumulation is fp32.
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cfloat>

namespace olmo_cpp {
namespace {

constexpr int kThreads = 512;

template <typename T>
__device__ __forceinline__ float ld(const T* p, int64_t i) { return static_cast<float>(p[i]); }

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
// Block-wide reduction; every thread gets the result.
template <bool IsMax>
__device__ float block_reduce(float v, float* sh) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = IsMax ? warp_max(v) : warp_sum(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  float r = (threadIdx.x < nw) ? sh[threadIdx.x] : (IsMax ? -FLT_MAX : 0.f);
  if (wid == 0) r = IsMax ? warp_max(r) : warp_sum(r);
  if (threadIdx.x == 0) sh[0] = r;
  __syncthreads();
  r = sh[0];
  __syncthreads();
  return r;
}

template <typename T>
__global__ void rowstats_kernel(const T* __restrict__ y, const T* __restrict__ p, const float* __restrict__ a,
                                const float* __restrict__ s, const int64_t* __restrict__ labels, int64_t V,
                                float* __restrict__ lse, float* __restrict__ picked) {
  __shared__ float sh[32];
  const int64_t row = blockIdx.x;
  const T* yr = y + row * V;
  const T* pr = p + row * V;
  const float av = *a, sv = s[row];
  // Online max / sum-exp per thread, then merged across the block.
  float m = -FLT_MAX, l = 0.f;
  for (int64_t v = threadIdx.x; v < V; v += blockDim.x) {
    const float x = sv * (av * ld(yr, v) + ld(pr, v));
    if (x > m) { l = l * __expf(m - x) + 1.f; m = x; }
    else l += __expf(x - m);
  }
  const float M = block_reduce<true>(m, sh);
  const float L = block_reduce<false>(m == -FLT_MAX ? 0.f : l * __expf(m - M), sh);
  if (threadIdx.x == 0) {
    lse[row] = M + logf(L);
    const int64_t lab = labels[row];
    picked[row] = (lab >= 0 && lab < V) ? sv * (av * ld(yr, lab) + ld(pr, lab)) : 0.f;
  }
}

template <typename T>
__global__ void grad_kernel(const T* __restrict__ y, const T* __restrict__ p, const float* __restrict__ a,
                            const float* __restrict__ coef, const float* __restrict__ s,
                            const float* __restrict__ lse, const int64_t* __restrict__ labels,
                            const float* __restrict__ w, int64_t V, T* __restrict__ gs,
                            float* __restrict__ dy, float* __restrict__ gz, float* __restrict__ gsy) {
  __shared__ float sh[32];
  const int64_t row = blockIdx.x;
  const T* yr = y + row * V;
  const T* pr = p + row * V;
  float* dyr = dy + row * V;
  T* gsr = gs ? gs + row * V : nullptr;
  const float av = *a, cv = *coef, sv = s[row], L = lse[row], wv = w[row];
  const int64_t lab = labels[row];
  float acc_gz = 0.f, acc_gsy = 0.f;
  for (int64_t v = threadIdx.x; v < V; v += blockDim.x) {
    const float yv = ld(yr, v);
    const float z = av * yv + ld(pr, v);
    float G = __expf(sv * z - L);
    if (v == lab) G -= 1.f;
    G *= wv;
    const float g_s = G * sv;
    if (gsr) gsr[v] = static_cast<T>(g_s);
    dyr[v] += cv * g_s;
    acc_gz += G * z;
    acc_gsy += g_s * yv;
  }
  const float tz = block_reduce<false>(acc_gz, sh);
  const float ty = block_reduce<false>(acc_gsy, sh);
  if (threadIdx.x == 0) {
    gz[row] = tz;
    gsy[row] = ty;
  }
}

}  // namespace

void sl_mtp_ce_rowstats_cuda(const torch::Tensor& y, const torch::Tensor& p, const torch::Tensor& a,
                             const torch::Tensor& s, const torch::Tensor& labels, torch::Tensor& lse,
                             torch::Tensor& picked) {
  TORCH_CHECK(y.is_contiguous() && p.is_contiguous() && y.sizes() == p.sizes(), "rowstats: y, p must match");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32 && s.scalar_type() == torch::kFloat32, "rowstats: fp32 a, s");
  c10::cuda::CUDAGuard guard(y.device());
  const int64_t rows = y.size(0), V = y.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, y.scalar_type(), "sl_mtp_rowstats", [&] {
    rowstats_kernel<scalar_t><<<rows, kThreads, 0, stream>>>(
        y.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), a.data_ptr<float>(), s.data_ptr<float>(),
        labels.data_ptr<int64_t>(), V, lse.data_ptr<float>(), picked.data_ptr<float>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sl_mtp_ce_grad_cuda(const torch::Tensor& y, const torch::Tensor& p, const torch::Tensor& a,
                         const torch::Tensor& coef, const torch::Tensor& s, const torch::Tensor& lse,
                         const torch::Tensor& labels, const torch::Tensor& w, torch::Tensor* gs,
                         torch::Tensor& dy, torch::Tensor& gz, torch::Tensor& gsy) {
  TORCH_CHECK(dy.scalar_type() == torch::kFloat32 && dy.is_contiguous(), "grad: dy must be contiguous fp32");
  c10::cuda::CUDAGuard guard(y.device());
  const int64_t rows = y.size(0), V = y.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, y.scalar_type(), "sl_mtp_grad", [&] {
    grad_kernel<scalar_t><<<rows, kThreads, 0, stream>>>(
        y.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), a.data_ptr<float>(), coef.data_ptr<float>(),
        s.data_ptr<float>(), lse.data_ptr<float>(), labels.data_ptr<int64_t>(), w.data_ptr<float>(), V,
        gs ? gs->data_ptr<scalar_t>() : nullptr, dy.data_ptr<float>(), gz.data_ptr<float>(),
        gsy.data_ptr<float>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace olmo_cpp
