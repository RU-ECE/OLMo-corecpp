#pragma once

#include "zwt/core/stream.hpp"
#include "zwt/layers/parameter.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace zwt::dist {

// Data-parallel gradient synchronization with bucketing.
//
// Each rank owns a full model copy and computes gradients on its own shard
// of the global batch. After backward, every gradient tensor must be
// all-reduced across ranks before the optimizer step sees it.
//
// Doing one all-reduce per parameter is the wrong shape: a 1B model has
// hundreds of parameters, each all-reduce incurs a launch + ring-trip cost,
// and small allreduces are latency-bound on NCCL. Instead we:
//   1. Bucket parameters by fill order (reverse of forward iteration order
//      — approximates the order gradients finish in the backward pass).
//   2. Size each bucket at a fixed byte budget (default 25 MiB).
//   3. As soon as all params in a bucket have their grad ready, concatenate
//      (gather-copy) into a contiguous fp32 buffer and fire a single
//      ncclAllReduce on that buffer on a side stream.
//   4. On the next step's forward, wait on the side stream's event before
//      reading grads from the optimizer — this overlaps all-reduce with
//      compute of the subsequent layer's backward.
//
// This class owns the bucketing logic + host-side scheduling. The actual
// NCCL call is parameterized via an AllReduceFn callback so that (a) the
// NCCL dep stays at the boundary and (b) CPU builds and single-rank unit
// tests can exercise bucketing without linking NCCL.
class BucketManager {
 public:
  // params: in the order the optimizer sees them (parameter registration
  //         order). Bucketing is assigned in reverse so grads that finish
  //         earliest in backward go into buckets that fire earliest.
  // bucket_bytes: target bucket size. Actual buckets may exceed this by one
  //         parameter — we never split a single parameter across buckets.
  // allreduce: callback to invoke once a bucket is ready.
  //            Signature: (fp32 buffer, element count, stream) -> void.
  BucketManager(const std::vector<Parameter*>& params,
                size_t bucket_bytes,
                int world_size);

  // Hand-off a parameter whose grad is now fully populated. When every
  // parameter in a bucket has been marked ready, `allreduce(buf, n, stream)`
  // fires. If the grad was never zero'd this step, calling mark_ready()
  // exactly once per optimizer-visible parameter is still correct.
  //
  // The AllReduceFn callback is responsible for a sum-allreduce, then
  // dividing by world_size. The manager scales the reduced result by
  // 1/world_size itself before writing grads back into Parameter::grad.
  using AllReduceFn = std::function<void(float* buf, size_t n, StreamHandle s)>;

  void set_allreduce(AllReduceFn fn) { allreduce_ = std::move(fn); }

  // Indicate that grad for parameter i is ready. Fires all-reduce for any
  // bucket that becomes complete.
  void mark_ready(int param_index, StreamHandle s);

  // Wait for all in-flight all-reduces, scatter results back into each
  // parameter's grad, and reset the bucket state for the next step.
  void finalize();

  // Reset ready-state. Call at the start of each step (or the end of
  // finalize — same effect).
  void begin_step();

  int num_buckets() const { return static_cast<int>(buckets_.size()); }
  size_t bucket_bytes() const { return bucket_bytes_; }

 private:
  struct Bucket {
    std::vector<int> param_ids;   // indexes into params_
    size_t total_floats = 0;      // sum of numel() across params
    std::vector<float> staging;   // host-side fp32 buffer; device version
                                  // will come with NCCL hookup
    int remaining = 0;            // params not yet marked ready
    bool fired = false;
  };

  const std::vector<Parameter*>& params_;
  size_t bucket_bytes_;
  int    world_size_;
  std::vector<int> param_to_bucket_;  // param index -> bucket index
  std::vector<Bucket> buckets_;
  AllReduceFn allreduce_;
};

// Create a BucketManager from the parameters a Transformer exposes. Ordering
// matches the collect_params() traversal; bucketing runs in reverse so early
// buckets hold the output layer's params (which finish first in backward).
BucketManager make_ddp(const std::vector<Parameter*>& params,
                       size_t bucket_bytes = size_t(25) << 20,
                       int world_size = 1);

}  // namespace zwt::dist
