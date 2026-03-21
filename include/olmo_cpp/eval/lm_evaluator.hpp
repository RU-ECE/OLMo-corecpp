#pragma once

#include "olmo_cpp/eval/evaluator.hpp"
#include <string>
#include <vector>

namespace olmo_cpp {

/// Language model evaluator: perplexity + accuracy on validation data
class LMEvaluator : public Evaluator {
 public:
  LMEvaluator(const std::string& data_path, int64_t seq_len,
              int64_t batch_size = 8, int64_t max_batches = -1);

  MetricMap evaluate(torch::nn::Module& model, torch::Device device) override;
  std::string name() const override { return "lm_eval"; }

 private:
  std::string data_path_;
  int64_t seq_len_, batch_size_, max_batches_;
};

/// Multiple-choice downstream task evaluator
class DownstreamEvaluator : public Evaluator {
 public:
  DownstreamEvaluator(const std::string& task_name, const std::string& data_path,
                      int64_t max_examples = -1);

  MetricMap evaluate(torch::nn::Module& model, torch::Device device) override;
  std::string name() const override { return name_; }

 private:
  std::string name_;
  struct Example {
    std::string context;
    std::vector<std::string> choices;
    int64_t label;
  };
  std::vector<Example> examples_;
};

}  // namespace olmo_cpp
