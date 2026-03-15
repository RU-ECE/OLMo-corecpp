#include "olmo_cpp/eval/lm_evaluator.hpp"
#include "olmo_cpp/data/token_dataset.hpp"
#include <iostream>
#include <fstream>
#include <string>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// LMEvaluator
// ---------------------------------------------------------------------------

LMEvaluator::LMEvaluator(const std::string& data_path, int64_t seq_len,
                          int64_t batch_size, int64_t max_batches)
    : data_path_(data_path), seq_len_(seq_len),
      batch_size_(batch_size), max_batches_(max_batches) {}

MetricMap LMEvaluator::evaluate(torch::nn::Module& /*model*/, torch::Device /*device*/) {
  // NOTE: To properly evaluate, callers should use the concrete Transformer type.
  // This base implementation loads data and reports dataset stats.
  // The full eval loop is in train.cpp where we have access to the Transformer type.
  MetricMap metrics;
  metrics["data_path_set"] = 1.0;
  metrics["seq_len"] = static_cast<double>(seq_len_);
  metrics["batch_size"] = static_cast<double>(batch_size_);
  return metrics;
}

// ---------------------------------------------------------------------------
// DownstreamEvaluator
// ---------------------------------------------------------------------------

DownstreamEvaluator::DownstreamEvaluator(const std::string& task_name,
                                          const std::string& data_path,
                                          int64_t max_examples)
    : name_(task_name) {
#ifdef HAS_NLOHMANN_JSON
  std::ifstream in(data_path);
  if (!in.is_open()) return;
  std::string line;
  int64_t count = 0;
  while (std::getline(in, line)) {
    if (max_examples > 0 && count >= max_examples) break;
    auto j = nlohmann::json::parse(line);
    Example ex;
    ex.context = j.value("context", std::string(""));
    ex.label = j.value("label", int64_t(0));
    if (j.contains("choices")) {
      for (const auto& c : j["choices"]) {
        ex.choices.push_back(c.get<std::string>());
      }
    }
    examples_.push_back(std::move(ex));
    count++;
  }
#else
  (void)data_path;
  (void)max_examples;
#endif
}

MetricMap DownstreamEvaluator::evaluate(torch::nn::Module& /*model*/,
                                         torch::Device /*device*/) {
  MetricMap metrics;
  metrics["num_examples"] = static_cast<double>(examples_.size());
  metrics["accuracy"] = 0.0;  // requires tokenizer integration
  return metrics;
}

}  // namespace olmo_cpp
