#include "zwt/optim/grad_clip.hpp"
#include "zwt/core/stream.hpp"

#include <cmath>
#include <cstdint>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace zwt::optim {

#ifdef USE_CUDA
namespace k {
void sumsq_fp32_many(float** ptrs, const int64_t* sizes, int n_tensors,
                     float* out, cudaStream_t s);
void scale_fp32_many(float** ptrs, const int64_t* sizes, int n_tensors,
                     float alpha, cudaStream_t s);
}  // namespace k
#endif

float clip_grad_norm(const std::vector<Parameter*>& params, float max_norm) {
  if (params.empty()) return 0.f;
  bool on_cuda = params.front()->value.device().is_cuda();

  if (on_cuda) {
#ifdef USE_CUDA
    // Build pointer + size arrays on host, upload once, reduce once, scale once.
    const int n = static_cast<int>(params.size());
    std::vector<float*>  p_h(n);
    std::vector<int64_t> s_h(n);
    for (int i = 0; i < n; ++i) {
      p_h[i] = params[i]->grad.as<float>();
      s_h[i] = params[i]->value.numel();
    }
    Device dev = params.front()->value.device();
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(compute_stream(dev).handle);

    // Allocate device scratch: pointer array + size array + scalar result.
    float**  d_ptrs = nullptr;
    int64_t* d_sizes = nullptr;
    float*   d_result = nullptr;
    cudaMallocAsync(&d_ptrs,   sizeof(float*)  * n, stream);
    cudaMallocAsync(&d_sizes,  sizeof(int64_t) * n, stream);
    cudaMallocAsync(&d_result, sizeof(float), stream);
    cudaMemcpyAsync(d_ptrs,  p_h.data(), sizeof(float*)  * n, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sizes, s_h.data(), sizeof(int64_t) * n, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_result, 0, sizeof(float), stream);

    k::sumsq_fp32_many(d_ptrs, d_sizes, n, d_result, stream);

    // Pull the norm to host (sync point — one per step, ~5us).
    float sumsq = 0.f;
    cudaMemcpyAsync(&sumsq, d_result, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    float norm = std::sqrt(sumsq);

    if (max_norm > 0.f && norm > max_norm) {
      float scale = max_norm / (norm + 1e-6f);
      k::scale_fp32_many(d_ptrs, d_sizes, n, scale, stream);
    }

    cudaFreeAsync(d_ptrs,   stream);
    cudaFreeAsync(d_sizes,  stream);
    cudaFreeAsync(d_result, stream);
    return norm;
#endif
  }

  // CPU reference.
  double sumsq = 0.0;
  for (auto* p : params) {
    const float* g = p->grad.as<float>();
    const int64_t n = p->value.numel();
    for (int64_t i = 0; i < n; ++i) sumsq += double(g[i]) * double(g[i]);
  }
  float norm = std::sqrt(float(sumsq));
  if (max_norm > 0.f && norm > max_norm) {
    float scale = max_norm / (norm + 1e-6f);
    for (auto* p : params) {
      float* g = p->grad.as<float>();
      const int64_t n = p->value.numel();
      for (int64_t i = 0; i < n; ++i) g[i] *= scale;
    }
  }
  return norm;
}

}  // namespace zwt::optim
