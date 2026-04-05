// Fused RMSNorm CUDA kernel - optimized for H100 (sm_90)
// Supports FP32 and BF16 with internal FP32 accumulation for numerical stability
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ float warpReduceSum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warpReduceSum(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
  if (wid == 0) val = warpReduceSum(val);
  return val;
}

// Templated RMSNorm kernel: works with float and __nv_bfloat16
// Internal math always in FP32 for stability
template <typename T>
__global__ void rms_norm_fwd_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const T* row_x = x + row * dim;
  T* row_out = out + row * dim;

  // Sum of squares in FP32
  float sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = static_cast<float>(row_x[i]);
    sum_sq += v * v;
  }

  sum_sq = blockReduceSum(sum_sq);

  __shared__ float s_rms_scale;
  if (threadIdx.x == 0) {
    s_rms_scale = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
  }
  __syncthreads();

  float scale = s_rms_scale;

  // Normalize and write
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = static_cast<float>(row_x[i]) * scale;
    if (weight) v *= static_cast<float>(weight[i]);
    row_out[i] = static_cast<T>(v);
  }
}

// Vectorized FP32 specialization (4 floats at a time)
__global__ void rms_norm_fwd_f32_vec(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const float* row_x = x + row * dim;
  float* row_out = out + row * dim;

  float sum_sq = 0.0f;
  int64_t vec_dim = dim / 4;
  const float4* x4 = reinterpret_cast<const float4*>(row_x);

  for (int64_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    float4 v = x4[i];
    sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
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

// Fused residual + RMSNorm (templated for FP32/BF16)
template <typename T>
__global__ void residual_rms_norm_fwd_kernel(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ out,
    T* __restrict__ residual_out,
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const T* row_x = x + row * dim;
  const T* row_res = residual + row * dim;
  T* row_out = out + row * dim;
  T* row_res_out = residual_out + row * dim;

  float sum_sq = 0.0f;

  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float h = static_cast<float>(row_x[i]) + static_cast<float>(row_res[i]);
    row_res_out[i] = static_cast<T>(h);
    sum_sq += h * h;
  }

  sum_sq = blockReduceSum(sum_sq);

  __shared__ float s_rms_scale;
  if (threadIdx.x == 0) {
    s_rms_scale = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
  }
  __syncthreads();

  float scale = s_rms_scale;

  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float h = static_cast<float>(row_res_out[i]) * scale;
    if (weight) h *= static_cast<float>(weight[i]);
    row_out[i] = static_cast<T>(h);
  }
}

}  // namespace

// C++ entry points with dtype dispatch

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

  int threads = (dim <= 256) ? 128 : 256;

  if (x.scalar_type() == torch::kFloat32) {
    rms_norm_fwd_f32_vec<<<rows, threads>>>(
        x_contig.data_ptr<float>(),
        weight.has_value() ? weight->contiguous().data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        dim,
        static_cast<float>(eps));
  } else if (x.scalar_type() == torch::kBFloat16) {
    auto w_contig = weight.has_value() ? weight->contiguous().to(torch::kBFloat16) : torch::Tensor();
    rms_norm_fwd_kernel<__nv_bfloat16><<<rows, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_contig.data_ptr<at::BFloat16>()),
        w_contig.defined() ? reinterpret_cast<const __nv_bfloat16*>(w_contig.data_ptr<at::BFloat16>()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        dim,
        static_cast<float>(eps));
  } else {
    TORCH_CHECK(false, "rms_norm CUDA: unsupported dtype (need fp32 or bf16)");
  }
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

  if (x.scalar_type() == torch::kFloat32) {
    residual_rms_norm_fwd_kernel<float><<<rows, threads>>>(
        x_contig.data_ptr<float>(),
        res_contig.data_ptr<float>(),
        weight.has_value() ? weight->contiguous().data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        residual_out.data_ptr<float>(),
        dim,
        static_cast<float>(eps));
  } else if (x.scalar_type() == torch::kBFloat16) {
    auto w_contig = weight.has_value() ? weight->contiguous().to(torch::kBFloat16) : torch::Tensor();
    residual_rms_norm_fwd_kernel<__nv_bfloat16><<<rows, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_contig.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(res_contig.data_ptr<at::BFloat16>()),
        w_contig.defined() ? reinterpret_cast<const __nv_bfloat16*>(w_contig.data_ptr<at::BFloat16>()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
        dim,
        static_cast<float>(eps));
  } else {
    TORCH_CHECK(false, "residual_rms_norm CUDA: unsupported dtype");
  }
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
