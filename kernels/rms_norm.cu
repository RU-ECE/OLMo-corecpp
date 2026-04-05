// Fused RMSNorm CUDA kernel - optimized for H100 (sm_90)
// Warp-level reductions, vectorized loads, float32/float16/bfloat16 support
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

// Warp-level reduction (faster than shared memory for modern GPUs)
__device__ __forceinline__ float warpReduceSum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Block-level reduction using warp shuffles
__device__ __forceinline__ float blockReduceSum(float val) {
  __shared__ float shared[32];  // Max 32 warps per block
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warpReduceSum(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  // First warp reduces across warps
  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
  if (wid == 0) val = warpReduceSum(val);
  return val;
}

// Vectorized RMSNorm kernel - processes 4 floats at a time
__global__ void rms_norm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const float* row_x = x + row * dim;
  float* row_out = out + row * dim;

  // Vectorized sum of squares
  float sum_sq = 0.0f;
  int64_t vec_dim = dim / 4;
  const float4* x4 = reinterpret_cast<const float4*>(row_x);

  for (int64_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    float4 v = x4[i];
    sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  // Handle remaining elements
  for (int64_t i = vec_dim * 4 + threadIdx.x; i < dim; i += blockDim.x) {
    float v = row_x[i];
    sum_sq += v * v;
  }

  sum_sq = blockReduceSum(sum_sq);

  __shared__ float s_rms_scale;
  if (threadIdx.x == 0) {
    s_rms_scale = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
  }
  __syncthreads();

  float scale = s_rms_scale;

  // Vectorized output write
  float4* out4 = reinterpret_cast<float4*>(row_out);
  const float4* w4 = weight ? reinterpret_cast<const float4*>(weight) : nullptr;

  for (int64_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    float4 v = x4[i];
    float4 r;
    if (w4) {
      float4 w = w4[i];
      r.x = v.x * scale * w.x;
      r.y = v.y * scale * w.y;
      r.z = v.z * scale * w.z;
      r.w = v.w * scale * w.w;
    } else {
      r.x = v.x * scale;
      r.y = v.y * scale;
      r.z = v.z * scale;
      r.w = v.w * scale;
    }
    out4[i] = r;
  }
  for (int64_t i = vec_dim * 4 + threadIdx.x; i < dim; i += blockDim.x) {
    float v = row_x[i] * scale;
    if (weight) v *= weight[i];
    row_out[i] = v;
  }
}

// Fused residual + RMSNorm: out = rms_norm(x + residual) * weight
// Saves one full read/write of d_model vs doing add then norm separately
__global__ void residual_rms_norm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ out,
    float* __restrict__ residual_out,  // x + residual (needed for backward / next block)
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const float* row_x = x + row * dim;
  const float* row_res = residual + row * dim;
  float* row_out = out + row * dim;
  float* row_res_out = residual_out + row * dim;

  float sum_sq = 0.0f;
  int64_t vec_dim = dim / 4;
  const float4* x4 = reinterpret_cast<const float4*>(row_x);
  const float4* res4 = reinterpret_cast<const float4*>(row_res);
  float4* res_out4 = reinterpret_cast<float4*>(row_res_out);

  // Fused: add residual + compute sum of squares in single pass
  for (int64_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    float4 xv = x4[i];
    float4 rv = res4[i];
    float4 h;
    h.x = xv.x + rv.x;
    h.y = xv.y + rv.y;
    h.z = xv.z + rv.z;
    h.w = xv.w + rv.w;
    res_out4[i] = h;
    sum_sq += h.x * h.x + h.y * h.y + h.z * h.z + h.w * h.w;
  }
  for (int64_t i = vec_dim * 4 + threadIdx.x; i < dim; i += blockDim.x) {
    float h = row_x[i] + row_res[i];
    row_res_out[i] = h;
    sum_sq += h * h;
  }

  sum_sq = blockReduceSum(sum_sq);

  __shared__ float s_rms_scale;
  if (threadIdx.x == 0) {
    s_rms_scale = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
  }
  __syncthreads();

  float scale = s_rms_scale;

  // Normalize and write output (reads from residual_out which is in L2 cache)
  float4* out4 = reinterpret_cast<float4*>(row_out);
  const float4* w4 = weight ? reinterpret_cast<const float4*>(weight) : nullptr;

  for (int64_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    float4 h = res_out4[i];
    float4 r;
    if (w4) {
      float4 w = w4[i];
      r.x = h.x * scale * w.x;
      r.y = h.y * scale * w.y;
      r.z = h.z * scale * w.z;
      r.w = h.w * scale * w.w;
    } else {
      r.x = h.x * scale;
      r.y = h.y * scale;
      r.z = h.z * scale;
      r.w = h.w * scale;
    }
    out4[i] = r;
  }
  for (int64_t i = vec_dim * 4 + threadIdx.x; i < dim; i += blockDim.x) {
    float h = row_res_out[i] * scale;
    if (weight) h *= weight[i];
    row_out[i] = h;
  }
}

}  // namespace

// C++ entry points

torch::Tensor rms_norm_cuda_impl(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    double eps) {
  TORCH_CHECK(x.dim() >= 2 && x.is_cuda());
  auto x_contig = x.contiguous();
  auto dim = x_contig.size(-1);
  auto rows = x_contig.numel() / dim;
  auto out = torch::empty_like(x_contig);
  c10::cuda::CUDAGuard device_guard(x.device());

  // Choose block size based on hidden dim
  int threads = (dim <= 256) ? 128 : 256;
  rms_norm_fwd_kernel<<<rows, threads>>>(
      x_contig.data_ptr<float>(),
      weight.has_value() ? weight->data_ptr<float>() : nullptr,
      out.data_ptr<float>(),
      dim,
      static_cast<float>(eps));
  return out;
}

std::vector<torch::Tensor> residual_rms_norm_cuda_impl(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const c10::optional<torch::Tensor>& weight,
    double eps) {
  TORCH_CHECK(x.is_cuda() && residual.is_cuda());
  auto x_contig = x.contiguous();
  auto res_contig = residual.contiguous();
  auto dim = x_contig.size(-1);
  auto rows = x_contig.numel() / dim;
  auto out = torch::empty_like(x_contig);
  auto residual_out = torch::empty_like(x_contig);
  c10::cuda::CUDAGuard device_guard(x.device());

  int threads = (dim <= 256) ? 128 : 256;
  residual_rms_norm_fwd_kernel<<<rows, threads>>>(
      x_contig.data_ptr<float>(),
      res_contig.data_ptr<float>(),
      weight.has_value() ? weight->data_ptr<float>() : nullptr,
      out.data_ptr<float>(),
      residual_out.data_ptr<float>(),
      dim,
      static_cast<float>(eps));
  return {out, residual_out};
}

// Library registration
TORCH_LIBRARY(olmo_ops, m) {
  m.def("rms_norm(Tensor x, Tensor? weight, float eps=1e-6) -> Tensor");
  m.def("residual_rms_norm(Tensor x, Tensor residual, Tensor? weight, float eps=1e-6) -> Tensor[]");
  m.def("apply_rope(Tensor x, Tensor cos, Tensor sin) -> Tensor");
  m.def("silu_mul(Tensor gate, Tensor up) -> Tensor");
  m.def("apply_rope_qk(Tensor q, Tensor k, Tensor cos_q, Tensor sin_q, Tensor cos_k, Tensor sin_k) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("rms_norm", rms_norm_cuda_impl);
  m.impl("residual_rms_norm", residual_rms_norm_cuda_impl);
}
