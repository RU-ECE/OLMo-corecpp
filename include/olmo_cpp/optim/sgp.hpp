#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>

namespace olmo_cpp {

/// Speculative Gradient Prediction (SGP) — skip backward passes by predicting
/// gradients from their recent history. Anchor-correct every K steps with a
/// real backward. Quality bounded by anchor frequency; speed from skip rate.
///
/// Usage: wrap the inner training loop. On each step, call should_skip_backward()
/// to decide whether to use predicted gradients or compute real ones. After a
/// real backward, call observe_real_gradients(). After a predicted step, call
/// apply_predicted_gradients().

struct SGPConfig {
  int64_t initial_k = 2;          // initial skip interval (predict K-1, anchor 1)
  int64_t max_k = 8;              // max skip interval
  int64_t min_k = 1;              // min skip interval (1 = no skipping)
  double grow_threshold = 0.1;    // grow K if prediction error < this
  double shrink_threshold = 0.3;  // shrink K if prediction error > this
  int64_t warmup_steps = 100;     // no skipping during warmup (gather statistics)
  int64_t min_param_numel = 1024; // only predict for params above this size
};

/// Abstract interface so v1 (linear) and v2 (rank-r subspace) can share a call site.
class ISGPPredictor {
 public:
  virtual ~ISGPPredictor() = default;
  virtual bool should_skip_backward(int64_t global_step) const = 0;
  virtual void observe_real_gradients() = 0;
  virtual void apply_predicted_gradients() = 0;
  virtual int64_t current_k() const = 0;
  virtual double last_prediction_error() const = 0;
  virtual int64_t skipped_steps() const = 0;
  virtual int64_t total_steps() const = 0;
  virtual double skip_rate() const = 0;
};

class SGPPredictor : public ISGPPredictor {
 public:
  SGPPredictor(const std::vector<torch::Tensor>& params, SGPConfig config = {});

  bool should_skip_backward(int64_t global_step) const override;
  void observe_real_gradients() override;
  void apply_predicted_gradients() override;

  int64_t current_k() const override { return k_; }
  double last_prediction_error() const override { return last_error_; }
  int64_t skipped_steps() const override { return skipped_; }
  int64_t total_steps() const override { return total_; }
  double skip_rate() const override { return total_ > 0 ? static_cast<double>(skipped_) / total_ : 0.0; }

 private:
  struct ParamState {
    torch::Tensor prev_grad;       // G_{t-1}
    torch::Tensor prev_prev_grad;  // G_{t-2}
    float alpha = 1.0f;            // linear predictor: G_pred = alpha * G_{t-1} + beta * G_{t-2}
    float beta = 0.0f;
    bool has_history = false;
    bool has_two_history = false;
  };

  void update_predictor_coefficients(ParamState& ps, const torch::Tensor& true_grad);

  SGPConfig config_;
  std::vector<torch::Tensor> params_;
  std::vector<ParamState> states_;
  int64_t k_;                      // current skip interval
  int64_t steps_since_anchor_;     // steps since last real backward
  double last_error_ = 0.0;
  int64_t skipped_ = 0;
  int64_t total_ = 0;
};

}  // namespace olmo_cpp
