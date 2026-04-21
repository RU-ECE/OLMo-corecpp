#pragma once

#include "zwt/core/device.hpp"
#include "zwt/core/stream.hpp"
#include "zwt/dist/ddp.hpp"
#include "zwt/layers/parameter.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zwt::dist {

// Overlap-aware DDP comm context.
//
// The layout:
//
//    compute_stream: ... bwd(layer_N) ── [grad_ready_event_N] ── bwd(layer_N-1) ...
//                                         │
//                                         ▼ record
//   comm_stream  :      allreduce(bucket_k, waits on ready_event)
//                                         │
//                                         ▼ record [bucket_done_event_k]
//    compute_stream                       ...wait on all bucket_done events
//                                           before opt.step()
//
// The bucket_done event is what opt.step() waits on to guarantee the fp32
// grad is fully reduced before AdamW reads it.
//
// The NCCL dep enters via a single function pointer type (IAllReduce). The
// `cuda/nccl_backend.cpp` translation unit (future) provides the real impl;
// CPU builds pass a memcpy-only stub that at least exercises the
// event/stream sequencing.
struct CommContext {
  int rank;
  int world_size;
  Device device;
  Stream comm_stream;          // side stream for reductions
  // Opaque backend handle. Under CUDA + NCCL this holds a ncclComm_t.
  // Under CPU builds it's null and the allreduce is a no-op.
  void* backend = nullptr;
};

// Wire a BucketManager into a CommContext. After this call:
//   * BucketManager::set_allreduce is set to a callback that records the
//     grad-ready event on compute_stream (supplied at call time), makes
//     comm_stream wait on it, invokes backend all-reduce on comm_stream,
//     then records bucket_done_event on comm_stream.
//   * The returned list of bucket_done Events must be waited on by
//     compute_stream before the next optimizer step reads Parameter::grad.
//
// `bucket_streams` is a ref out-parameter: caller owns the Event storage so
// it can be reused across steps without per-step allocation.
class OverlapHookup {
 public:
  OverlapHookup(BucketManager& mgr, CommContext ctx);

  // Call at the end of backward, before optimizer.step(). Makes the compute
  // stream wait on every fired bucket's done event, then calls
  // mgr.finalize().
  void sync_and_finalize();

  int num_buckets() const { return static_cast<int>(bucket_done_.size()); }

 private:
  BucketManager& mgr_;
  CommContext    ctx_;
  std::vector<Event> bucket_done_;  // one per bucket
};

// Thin no-op backend for CPU builds and unit tests — satisfies the contract
// but performs no reduction. Returns a CommContext whose `backend` is null.
CommContext make_loopback_ctx(Device dev);

}  // namespace zwt::dist
