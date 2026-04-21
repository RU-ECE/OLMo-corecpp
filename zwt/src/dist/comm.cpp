#include "zwt/dist/comm.hpp"

#include <stdexcept>

namespace zwt::dist {

OverlapHookup::OverlapHookup(BucketManager& mgr, CommContext ctx)
    : mgr_(mgr), ctx_(ctx) {
  bucket_done_.reserve(mgr_.num_buckets());
  for (int b = 0; b < mgr_.num_buckets(); ++b) {
    bucket_done_.push_back(Event::create(ctx_.device));
  }

  // Wire a callback that: records grad-ready on compute, makes comm_stream
  // wait on it, performs the reduction on comm_stream, and records
  // bucket_done_[b] on comm_stream. The mark_ready() call site already runs
  // on the compute stream, so we record the ready event on the
  // currently-active stream for the bucket device — passed in via
  // StreamHandle. The "which bucket just fired" is not surfaced by the
  // AllReduceFn signature yet (it only gets buf+n+stream). We chain the
  // ordering implicitly: all-reduces are issued sequentially on the same
  // comm_stream in bucket order, so the final bucket's done event
  // transitively ordered-after every earlier reduction.
  int* fire_idx = new int(0);
  mgr_.set_allreduce([this, fire_idx](float* /*buf*/, size_t /*n*/,
                                      StreamHandle compute_s) {
    // Host-side single-threaded: each mark_ready runs on one thread, so no
    // lock needed on fire_idx.
    Stream compute{ctx_.device, compute_s};
    Event  grad_ready = Event::create(ctx_.device);
    grad_ready.record(compute);
    grad_ready.wait(ctx_.comm_stream);

    // Real NCCL call would go here:
    //   ncclAllReduce(buf, buf, n, ncclFloat32, ncclSum, nccl_comm,
    //                 ctx_.comm_stream.handle);
    // Under CPU / loopback, it's a no-op.

    int b = (*fire_idx)++;
    if (b < static_cast<int>(bucket_done_.size())) {
      bucket_done_[b].record(ctx_.comm_stream);
    }
  });
  // Ownership leak of fire_idx is intentional for the duration of the
  // process — BucketManager's callback holds it. OverlapHookup lifetime is
  // tied to the training loop; cleaning it up at OverlapHookup dtor would
  // require tracking the std::function back, which is not worth the code.
}

void OverlapHookup::sync_and_finalize() {
  // Make the compute stream wait on every bucket_done event so AdamW reads
  // the fully-reduced grads.
  Stream compute = compute_stream(ctx_.device);
  for (auto& ev : bucket_done_) ev.wait(compute);
  mgr_.finalize();
}

CommContext make_loopback_ctx(Device dev) {
  CommContext c;
  c.rank        = 0;
  c.world_size  = 1;
  c.device      = dev;
  c.comm_stream = side_stream(dev);
  c.backend     = nullptr;
  return c;
}

}  // namespace zwt::dist
