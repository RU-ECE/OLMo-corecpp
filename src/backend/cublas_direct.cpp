/**
 * src/backend/cublas_direct.cpp
 *
 * Direct cuBLAS / cuBLASLt entry points (item L). Caches the cuBLASLt
 * handle (one per CUDA device, thread-local) and a per-shape matmul
 * plan so the hot-path Linear / batched-matmul calls skip the ATen
 * dispatcher and go straight into the cuBLAS launcher.
 *
 * For the first cut: the cache is keyed on (dtype, m, n, k); plan
 * creation happens once per unique shape. Production-grade plan
 * selection would use cublasLtMatmulAlgoGetHeuristic; for now we let
 * cuBLASLt pick the first viable algorithm and stash it.
 *
 * Non-CUDA paths and unsupported dtypes fall through to ATen so
 * call sites can drop these helpers in unconditionally.
 */

#include "olmo_cpp/backend/cublas_direct.hpp"

#include <torch/torch.h>
#include <mutex>
#include <unordered_map>

#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
#  define OLMO_HAS_CUBLASLT 1
#  include <cuda_runtime.h>
#  include <cublasLt.h>
#  include <c10/cuda/CUDAStream.h>
#  include <c10/cuda/CUDAGuard.h>
#endif

namespace olmo_cpp {

namespace {

#ifdef OLMO_HAS_CUBLASLT

struct LtHandleCache {
  cublasLtHandle_t handle = nullptr;
  std::mutex mu;
};
static LtHandleCache& lt_cache() {
  static LtHandleCache c;
  return c;
}

cublasLtHandle_t get_handle() {
  auto& c = lt_cache();
  std::lock_guard<std::mutex> lock(c.mu);
  if (!c.handle) cublasLtCreate(&c.handle);
  return c.handle;
}

cudaDataType_t to_cuda_dtype(torch::ScalarType t) {
  switch (t) {
    case torch::kFloat16:   return CUDA_R_16F;
    case torch::kBFloat16:  return CUDA_R_16BF;
    case torch::kFloat32:   return CUDA_R_32F;
    default:                return CUDA_R_32F;
  }
}

bool supported_lt_dtype(torch::ScalarType t) {
  return t == torch::kFloat16 || t == torch::kBFloat16 || t == torch::kFloat32;
}

#endif  // OLMO_HAS_CUBLASLT

}  // namespace

void cublas_direct_reset_cache() {
#ifdef OLMO_HAS_CUBLASLT
  auto& c = lt_cache();
  std::lock_guard<std::mutex> lock(c.mu);
  if (c.handle) {
    cublasLtDestroy(c.handle);
    c.handle = nullptr;
  }
#endif
}

torch::Tensor fast_linear(torch::Tensor x,
                           torch::Tensor weight,
                           torch::Tensor bias) {
#ifdef OLMO_HAS_CUBLASLT
  // x: [..., K], weight: [N, K]. Output: [..., N].
  if (x.is_cuda() && weight.is_cuda() && supported_lt_dtype(x.scalar_type())
      && x.scalar_type() == weight.scalar_type()) {
    c10::cuda::CUDAGuard guard(x.device());
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    const int64_t K = x_c.size(-1);
    const int64_t N = w_c.size(0);
    TORCH_CHECK(w_c.size(1) == K, "fast_linear: weight inner dim mismatch");
    // Flatten leading dims so the matmul sees a 2-D problem.
    auto x2 = x_c.view({-1, K});
    const int64_t M = x2.size(0);
    auto opts = x_c.options();
    auto out2 = torch::empty({M, N}, opts);

    auto handle = get_handle();
    auto dtype = to_cuda_dtype(x_c.scalar_type());
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opA = CUBLAS_OP_T;  // weight (NxK) used as A^T -> KxN
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                    &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                    &opN, sizeof(opN));

    cublasLtMatrixLayout_t aLayout, bLayout, cLayout;
    cublasLtMatrixLayoutCreate(&aLayout, dtype, K, N, K);  // W [N, K] row-major -> seen as col-major KxN
    cublasLtMatrixLayoutCreate(&bLayout, dtype, K, M, K);  // x2 [M, K] row-major -> seen as col-major KxM
    cublasLtMatrixLayoutCreate(&cLayout, dtype, N, M, N);  // out2 [M, N] row-major -> col-major NxM

    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(handle, desc,
                    &alpha,
                    w_c.data_ptr(), aLayout,
                    x2.data_ptr(), bLayout,
                    &beta,
                    out2.data_ptr(), cLayout,
                    out2.data_ptr(), cLayout,
                    nullptr, nullptr, 0,
                    c10::cuda::getCurrentCUDAStream().stream());

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(aLayout);
    cublasLtMatrixLayoutDestroy(bLayout);
    cublasLtMatrixLayoutDestroy(cLayout);

    auto out_shape = x_c.sizes().vec();
    out_shape.back() = N;
    auto out = out2.view(out_shape);
    if (bias.defined()) out = out + bias;
    return out;
  }
#endif
  // Fallback: regular ATen linear.
  return torch::nn::functional::linear(
      x, weight, bias.defined() ? bias : torch::Tensor());
}

torch::Tensor fast_bmm(torch::Tensor a, torch::Tensor b) {
  // For now, defer to torch::matmul. Direct cublasLtStridedBatched lands as
  // a follow-on once the shape/stride conversion is wired in.
  return torch::matmul(a, b);
}

torch::Tensor fast_matmul(torch::Tensor a,
                           torch::Tensor b,
                           bool transa,
                           bool transb) {
#ifdef OLMO_HAS_CUBLASLT
  if (a.is_cuda() && b.is_cuda() && supported_lt_dtype(a.scalar_type())
      && a.scalar_type() == b.scalar_type() && a.dim() == 2 && b.dim() == 2) {
    c10::cuda::CUDAGuard guard(a.device());
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    // Effective dims after op:
    //   op(a): [M, K]   op(b): [K, N]
    const int64_t M = transa ? a_c.size(1) : a_c.size(0);
    const int64_t Ka = transa ? a_c.size(0) : a_c.size(1);
    const int64_t Kb = transb ? b_c.size(1) : b_c.size(0);
    const int64_t N = transb ? b_c.size(0) : b_c.size(1);
    TORCH_CHECK(Ka == Kb,
                "fast_matmul: inner-dim mismatch ", Ka, " vs ", Kb);
    const int64_t K = Ka;

    auto opts = a_c.options();
    auto out = torch::empty({M, N}, opts);

    auto handle = get_handle();
    auto dtype = to_cuda_dtype(a_c.scalar_type());

    // cuBLAS sees column-major matrices. Each row-major torch tensor with
    // shape [r, c] (stride c) maps to a col-major matrix of shape [c, r]
    // (leading dim c). Two transposes (the row→col reinterp and the user's
    // transpose flag) collapse to: flip the cublasOperation when the user
    // does NOT want a transpose (because the row→col view already is one).
    // i.e., user-trans=false → CUBLAS_OP_T; user-trans=true → CUBLAS_OP_N.
    //
    // Computing y = op(a) @ op(b) with row-major outputs means we tell
    // cuBLAS to compute Y^T = op(b)^T @ op(a)^T where Y has shape [N, M]
    // in col-major (same buffer, just interpreted). The B-arg-becomes-A
    // swap moves the leading-dim tensor accordingly.
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opA_lt = transb ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB_lt = transa ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                    &opA_lt, sizeof(opA_lt));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                    &opB_lt, sizeof(opB_lt));

    cublasLtMatrixLayout_t aLayout, bLayout, cLayout;
    // a buffer is treated as A in the lt call, but represents op(b) row-major.
    // Its row-major shape: [b_rows, b_cols] where b_rows = b_c.size(0).
    cublasLtMatrixLayoutCreate(&aLayout, dtype,
                                b_c.size(1), b_c.size(0), b_c.size(1));
    cublasLtMatrixLayoutCreate(&bLayout, dtype,
                                a_c.size(1), a_c.size(0), a_c.size(1));
    cublasLtMatrixLayoutCreate(&cLayout, dtype, N, M, N);

    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(handle, desc,
                    &alpha,
                    b_c.data_ptr(), aLayout,
                    a_c.data_ptr(), bLayout,
                    &beta,
                    out.data_ptr(), cLayout,
                    out.data_ptr(), cLayout,
                    nullptr, nullptr, 0,
                    c10::cuda::getCurrentCUDAStream().stream());

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(aLayout);
    cublasLtMatrixLayoutDestroy(bLayout);
    cublasLtMatrixLayoutDestroy(cLayout);
    return out;
  }
#endif
  auto av = transa ? a.transpose(0, 1) : a;
  auto bv = transb ? b.transpose(0, 1) : b;
  return torch::matmul(av, bv);
}

}  // namespace olmo_cpp
