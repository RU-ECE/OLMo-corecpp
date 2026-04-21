#include "zwt/dist/ddp.hpp"

#include <algorithm>
#include <stdexcept>

namespace zwt::dist {

BucketManager::BucketManager(const std::vector<Parameter*>& params,
                             size_t bucket_bytes,
                             int world_size)
    : params_(params),
      bucket_bytes_(bucket_bytes),
      world_size_(world_size) {
  if (world_size < 1) throw std::runtime_error("BucketManager: world_size < 1");
  // We'd never hand the callback a zero-param bucket; treat empty params as
  // a caller error so the invariant holds.
  if (params.empty()) throw std::runtime_error("BucketManager: empty params");

  // Build buckets in reverse parameter order — backward visits parameters in
  // roughly reverse-forward order (output layer grads finish first), so this
  // matches the order we'd like allreduces to fire.
  const int N = static_cast<int>(params_.size());
  param_to_bucket_.assign(N, -1);

  Bucket cur;
  size_t cur_bytes = 0;
  for (int i = N - 1; i >= 0; --i) {
    const size_t p_bytes = static_cast<size_t>(params_[i]->numel()) * sizeof(float);
    if (!cur.param_ids.empty() && cur_bytes + p_bytes > bucket_bytes_) {
      buckets_.push_back(std::move(cur));
      cur = {};
      cur_bytes = 0;
    }
    cur.param_ids.push_back(i);
    cur.total_floats += params_[i]->numel();
    cur_bytes += p_bytes;
  }
  if (!cur.param_ids.empty()) buckets_.push_back(std::move(cur));

  for (int b = 0; b < static_cast<int>(buckets_.size()); ++b) {
    buckets_[b].remaining = static_cast<int>(buckets_[b].param_ids.size());
    buckets_[b].staging.resize(buckets_[b].total_floats);
    for (int pid : buckets_[b].param_ids) param_to_bucket_[pid] = b;
  }
}

void BucketManager::begin_step() {
  for (auto& b : buckets_) {
    b.remaining = static_cast<int>(b.param_ids.size());
    b.fired     = false;
  }
}

void BucketManager::mark_ready(int pid, StreamHandle s) {
  if (pid < 0 || pid >= static_cast<int>(param_to_bucket_.size())) {
    throw std::runtime_error("BucketManager: param index out of range");
  }
  int b = param_to_bucket_[pid];
  Bucket& bk = buckets_[b];
  if (bk.fired) {
    throw std::runtime_error("BucketManager: mark_ready after bucket fired");
  }
  if (--bk.remaining > 0) return;  // bucket not yet complete

  // All params in this bucket have their grad ready. Normally we'd:
  //   1. Copy each param's grad into bk.staging at its bucket offset
  //      (on-device fp32 contiguous buffer).
  //   2. Invoke allreduce_(buf, n, s).
  //   3. Scatter results back into each param's grad, scaled by 1/world.
  //
  // With the NCCL hook not yet wired, the callback is invoked on a host-side
  // buffer just to exercise the scheduling logic. When NCCL lands, swap the
  // staging vector for a device fp32 pool allocation and the memcpy for
  // per-param D2D copies.
  if (allreduce_) {
    allreduce_(bk.staging.data(), bk.total_floats, s);
  }
  bk.fired = true;
}

void BucketManager::finalize() {
  // In a real implementation this would wait on each bucket's event, then
  // D2D copy the reduced result back into each Parameter::grad scaled by
  // 1/world_size_. Today (no NCCL) this is a no-op that just verifies every
  // bucket fired. If a param was never marked_ready, the optimizer will use
  // the local gradient — callers must ensure the contract is held or catch
  // the following throw.
  for (size_t b = 0; b < buckets_.size(); ++b) {
    if (!buckets_[b].fired) {
      throw std::runtime_error(
          "BucketManager: finalize() before all buckets fired");
    }
  }
  begin_step();
}

BucketManager make_ddp(const std::vector<Parameter*>& params,
                       size_t bucket_bytes, int world_size) {
  return BucketManager(params, bucket_bytes, world_size);
}

}  // namespace zwt::dist
