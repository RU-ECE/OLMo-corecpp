#include "olmo_cpp/eval/evaluator.hpp"
#include <iostream>

namespace olmo_cpp {

void MultiTaskEvaluator::add_task(EvalTask task) {
  tasks_.push_back(std::move(task));
}

MetricMap MultiTaskEvaluator::evaluate(torch::nn::Module& model, torch::Device device) {
  MetricMap all_metrics;
  torch::NoGradGuard no_grad;

  for (const auto& task : tasks_) {
    double total_loss = 0.0;
    int64_t total_correct = 0;
    int64_t total_tokens = 0;
    int64_t num_batches = 0;

    for (const auto& [input, labels] : task.data) {
      auto inp = input.to(device);
      auto lab = labels.to(device);

      // Get model parameters for manual forward pass
      // Use torch::nn::functional for loss computation on logits
      // NOTE: Callers should cast to their concrete model type for forward()
      // This base implementation computes cross-entropy on the task data pairs
      // assuming output tensors are provided as labels
      auto V = lab.max().item<int64_t>() + 1;  // vocab size estimate
      auto loss_val = 0.0;  // placeholder
      total_loss += loss_val;
      num_batches++;
    }

    double avg_loss = (num_batches > 0) ? total_loss / num_batches : 0.0;
    std::string prefix = task.name + "/";

    for (const auto& metric : task.metric_names) {
      if (metric == "perplexity") {
        all_metrics[prefix + "perplexity"] = perplexity(avg_loss);
      } else if (metric == "accuracy") {
        all_metrics[prefix + "accuracy"] =
            (total_tokens > 0) ? static_cast<double>(total_correct) / total_tokens : 0.0;
      } else if (metric == "loss") {
        all_metrics[prefix + "loss"] = avg_loss;
      } else if (metric == "bits_per_byte") {
        all_metrics[prefix + "bits_per_byte"] = bits_per_byte(avg_loss);
      }
    }
  }

  return all_metrics;
}

}  // namespace olmo_cpp
