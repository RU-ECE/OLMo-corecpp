#pragma once
#include <torch/torch.h>
#include <deque>
#include <memory>

namespace olmo_cpp {

/// Skip-step optimizer wrapper: detects loss spikes and skips optimizer steps
class SkipStepOptimizer {
 public:
  SkipStepOptimizer(std::unique_ptr<torch::optim::Optimizer> inner,
                    double spike_threshold = 5.0,
                    int64_t window_size = 100);
  /// Returns true if step was taken, false if skipped due to spike
  bool step(float current_loss);
  void zero_grad();
  torch::optim::Optimizer& inner() { return *inner_; }
  int64_t skipped_steps() const { return skipped_steps_; }
 private:
  std::unique_ptr<torch::optim::Optimizer> inner_;
  double spike_threshold_;
  int64_t window_size_;
  std::deque<float> loss_history_;
  double running_mean_ = 0.0;
  double running_var_ = 1.0;
  int64_t skipped_steps_ = 0;
};

}  // namespace olmo_cpp
