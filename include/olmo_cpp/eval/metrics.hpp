#pragma once

#include <torch/torch.h>
#include <string>
#include <unordered_map>

namespace olmo_cpp {

using MetricMap = std::unordered_map<std::string, double>;

/// Compute perplexity from cross-entropy loss
double perplexity(double ce_loss);

/// Compute bits-per-byte from cross-entropy loss
double bits_per_byte(double ce_loss);

/// Accuracy: fraction of correct next-token predictions
double accuracy(const torch::Tensor& logits, const torch::Tensor& labels,
                int64_t ignore_index = -100);

/// Top-k accuracy
double top_k_accuracy(const torch::Tensor& logits, const torch::Tensor& labels,
                      int64_t k, int64_t ignore_index = -100);

/// F1 score for multi-class classification
double f1_score(const torch::Tensor& predictions, const torch::Tensor& labels,
                int64_t num_classes);

}  // namespace olmo_cpp
