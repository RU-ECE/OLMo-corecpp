#pragma once

#include <torch/torch.h>
#include "olmo_cpp/eval/metrics.hpp"
#include <string>
#include <vector>
#include <memory>

namespace olmo_cpp {

/// An evaluation task: dataset + metrics to compute
struct EvalTask {
  std::string name;
  std::vector<std::pair<torch::Tensor, torch::Tensor>> data;  // (input, label)
  std::vector<std::string> metric_names;  // e.g., {"perplexity", "accuracy"}
};

/// Base evaluator interface
class Evaluator {
 public:
  virtual ~Evaluator() = default;
  virtual MetricMap evaluate(torch::nn::Module& model, torch::Device device) = 0;
  virtual std::string name() const = 0;
};

/// Runs multiple eval tasks
class MultiTaskEvaluator : public Evaluator {
 public:
  void add_task(EvalTask task);
  MetricMap evaluate(torch::nn::Module& model, torch::Device device) override;
  std::string name() const override { return "multi_task"; }
 private:
  std::vector<EvalTask> tasks_;
};

}  // namespace olmo_cpp
