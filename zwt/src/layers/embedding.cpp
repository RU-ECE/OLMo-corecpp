#include "zwt/layers/embedding.hpp"
#include "zwt/core/stream.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#endif

namespace zwt {

namespace {

void normal_init(Tensor& w, float std_, uint64_t seed) {
  const size_t n = static_cast<size_t>(w.numel());
  uint64_t s = seed ? seed : 0xA5A5A5A5A5A5A5A5ULL;
  auto next = [&]() -> float {
    // Box-Muller via uniforms (good enough for init).
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x1 = static_cast<uint32_t>(s >> 33);
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x2 = static_cast<uint32_t>(s >> 33);
    float u1 = std::max((x1 / float(1u << 31) - 1.0f + 1.0f) * 0.5f, 1e-7f);
    float u2 = (x2 / float(1u << 31) - 1.0f);
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
  };

  std::vector<float> cpu(n);
  for (size_t i = 0; i < n; ++i) cpu[i] = next() * std_;

  if (w.dtype() == DType::F32) {
    Tensor host(cpu.data(), w.shape(), w.strides(), DType::F32, Device::cpu(),
                nullptr, cpu.size() * sizeof(float));
    copy(host, w);
  } else if (w.dtype() == DType::BF16) {
    std::vector<uint16_t> bf(n);
    for (size_t i = 0; i < n; ++i) {
      uint32_t u; std::memcpy(&u, &cpu[i], 4);
      uint32_t bias = 0x7FFF + ((u >> 16) & 1);
      bf[i] = static_cast<uint16_t>((u + bias) >> 16);
    }
    Tensor host(bf.data(), w.shape(), w.strides(), DType::BF16, Device::cpu(),
                nullptr, bf.size() * sizeof(uint16_t));
    copy(host, w);
  } else {
    throw std::runtime_error("Embedding::init: unsupported dtype");
  }
}

#ifdef USE_CUDA
// Gather kernel: for each (b,s,d), out[b,s,d] = W[ids[b,s], d].
__global__ void embed_gather_bf16(const int64_t* ids, const __nv_bfloat16* W,
                                  __nv_bfloat16* out, int64_t N, int64_t D) {
  int64_t i = blockIdx.x;
  if (i >= N) return;
  int64_t id = ids[i];
  const __nv_bfloat16* src = W + id * D;
  __nv_bfloat16* dst = out + i * D;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
}

// Scatter-add grad rows back into grad_W (fp32 grad buffer).
__global__ void embed_scatter_add(const int64_t* ids, const __nv_bfloat16* grad_y,
                                  float* grad_W, int64_t N, int64_t D) {
  int64_t i = blockIdx.x;
  if (i >= N) return;
  int64_t id = ids[i];
  const __nv_bfloat16* src = grad_y + i * D;
  float* dst = grad_W + id * D;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    atomicAdd(dst + d, __bfloat162float(src[d]));
  }
}
#endif

}  // namespace

Embedding::Embedding(int64_t vocab_size, int64_t d_model, DType dtype, Device device)
    : vocab_size_(vocab_size), d_model_(d_model) {
  weight_.name  = "weight";
  weight_.value = empty({vocab_size, d_model}, dtype, device);
  normal_init(weight_.value, 0.02f, 0xBEEFC0DEULL);
}

Tensor Embedding::forward(const Tensor& token_ids) {
  saved_ids_ = token_ids.view(token_ids.shape());

  Shape out_shape;
  out_shape.rank = token_ids.rank() + 1;
  for (int i = 0; i < token_ids.rank(); ++i) out_shape.dims[i] = token_ids.dim(i);
  out_shape.dims[token_ids.rank()] = d_model_;
  saved_out_shape_ = out_shape;

  Tensor out = empty_scratch(out_shape, weight_.value.dtype(), weight_.value.device());
  int64_t N = token_ids.numel();

  if (token_ids.device().is_cuda()) {
#ifdef USE_CUDA
    if (weight_.value.dtype() != DType::BF16) {
      throw std::runtime_error("Embedding CUDA path currently BF16 only");
    }
    embed_gather_bf16<<<static_cast<unsigned>(N), 256, 0,
        reinterpret_cast<cudaStream_t>(compute_stream(token_ids.device()).handle)>>>(
        token_ids.as<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(weight_.value.data()),
        reinterpret_cast<__nv_bfloat16*>(out.data()),
        N, d_model_);
    return out;
#else
    throw std::runtime_error("Embedding: CUDA path requested on CPU-only build");
#endif
  }

  // CPU path: f32 only.
  if (weight_.value.dtype() != DType::F32) {
    throw std::runtime_error("Embedding CPU path: F32 only");
  }
  const int64_t* ids = token_ids.as<int64_t>();
  const float*   W   = weight_.value.as<float>();
  float*         O   = out.as<float>();
  for (int64_t i = 0; i < N; ++i) {
    int64_t id = ids[i];
    if (id < 0 || id >= vocab_size_) throw std::runtime_error("Embedding: id OOB");
    std::memcpy(O + i * d_model_, W + id * d_model_, d_model_ * sizeof(float));
  }
  return out;
}

Tensor Embedding::backward(const Tensor& grad_y) {
  weight_.ensure_grad();
  int64_t N = grad_y.numel() / d_model_;

  if (grad_y.device().is_cuda()) {
#ifdef USE_CUDA
    embed_scatter_add<<<static_cast<unsigned>(N), 256, 0,
        reinterpret_cast<cudaStream_t>(compute_stream(grad_y.device()).handle)>>>(
        saved_ids_.as<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(grad_y.data()),
        weight_.grad.as<float>(),
        N, d_model_);
#endif
  } else {
    const int64_t* ids = saved_ids_.as<int64_t>();
    const float*   gy  = grad_y.as<float>();
    float*         gW  = weight_.grad.as<float>();
    for (int64_t i = 0; i < N; ++i) {
      int64_t id = ids[i];
      for (int64_t d = 0; d < d_model_; ++d) {
        gW[id * d_model_ + d] += gy[i * d_model_ + d];
      }
    }
  }

  // Embedding has no upstream grad_x (input is integer ids).
  return Tensor{};
}

void Embedding::collect_params(std::vector<Parameter*>& out) {
  out.push_back(&weight_);
}

}  // namespace zwt
