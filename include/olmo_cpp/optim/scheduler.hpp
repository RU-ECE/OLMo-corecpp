#pragma once
#include <torch/torch.h>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace olmo_cpp {

/// Base LR scheduler interface
class LRScheduler {
 public:
  virtual ~LRScheduler() = default;
  virtual double get_lr(int64_t step, int64_t total_steps) const = 0;
  void apply(torch::optim::Optimizer& optimizer, int64_t step, int64_t total_steps) const;
};

/// Constant LR
class ConstantLR : public LRScheduler {
 public:
  explicit ConstantLR(double lr) : lr_(lr) {}
  double get_lr(int64_t step, int64_t total_steps) const override { return lr_; }
 private:
  double lr_;
};

/// Linear decay from base_lr to 0
class LinearDecayLR : public LRScheduler {
 public:
  LinearDecayLR(double base_lr, int64_t warmup_steps = 0, double min_lr = 0.0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, min_lr_;
  int64_t warmup_steps_;
};

/// Cosine annealing with optional warmup
class CosineAnnealingLR : public LRScheduler {
 public:
  CosineAnnealingLR(double base_lr, int64_t warmup_steps = 0, double min_lr = 0.0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, min_lr_;
  int64_t warmup_steps_;
};

/// Exponential decay
class ExponentialDecayLR : public LRScheduler {
 public:
  ExponentialDecayLR(double base_lr, double gamma, int64_t warmup_steps = 0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, gamma_;
  int64_t warmup_steps_;
};

/// Inverse square root schedule (used in original Transformer paper)
class InverseSqrtLR : public LRScheduler {
 public:
  InverseSqrtLR(double base_lr, int64_t warmup_steps);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_;
  int64_t warmup_steps_;
};

/// Polynomial decay
class PolynomialDecayLR : public LRScheduler {
 public:
  PolynomialDecayLR(double base_lr, double power = 1.0, int64_t warmup_steps = 0, double min_lr = 0.0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, power_, min_lr_;
  int64_t warmup_steps_;
};

/// Step decay: multiply lr by gamma every step_size steps
class StepDecayLR : public LRScheduler {
 public:
  StepDecayLR(double base_lr, double gamma, int64_t step_size, int64_t warmup_steps = 0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, gamma_;
  int64_t step_size_, warmup_steps_;
};

/// Warmup-Stable-Decay (WSD) schedule used by OLMo
class WSDLR : public LRScheduler {
 public:
  WSDLR(double base_lr, int64_t warmup_steps, int64_t stable_steps, double min_lr = 0.0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, min_lr_;
  int64_t warmup_steps_, stable_steps_;
};

/// Leslie Smith LR range test (2017): exponentially sweep LR from start_lr to
/// end_lr across total_steps. Run for a short pretraining warmup, plot loss vs
/// lr, and pick the LR just before loss explodes (typically 1/10th of the lr
/// that gave the steepest descent).
///
/// Usage: set scheduler = lr_range_test, set [training] lr = <end_lr> (upper
/// bound, e.g. 1.0), min_lr = <start_lr> (lower bound, e.g. 1e-8), steps = 100.
/// Keep log_interval = 1 so every step prints a point.
class LRRangeTest : public LRScheduler {
 public:
  LRRangeTest(double start_lr, double end_lr);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double log_start_, log_end_;
};

/// Custom schedule from a user-provided function
class CustomLR : public LRScheduler {
 public:
  using ScheduleFn = std::function<double(int64_t step, int64_t total_steps)>;
  explicit CustomLR(ScheduleFn fn) : fn_(std::move(fn)) {}
  double get_lr(int64_t step, int64_t total_steps) const override { return fn_(step, total_steps); }
 private:
  ScheduleFn fn_;
};

/// Factory to create schedulers by name
std::unique_ptr<LRScheduler> create_scheduler(
    const std::string& name, double base_lr, int64_t warmup_steps = 0,
    double min_lr = 0.0, double gamma = 0.1, int64_t step_size = 1000,
    double power = 1.0, int64_t stable_steps = 0);

}  // namespace olmo_cpp
