#include "olmo_cpp/eval/metrics.hpp"
#include <cmath>

namespace olmo_cpp {

double perplexity(double ce_loss) {
  return std::exp(ce_loss);
}

double bits_per_byte(double ce_loss) {
  // bits_per_byte = ce_loss / ln(2) / chars_per_token
  // Using standard ~3.5 chars/token for English text
  return ce_loss / std::log(2.0);
}

double accuracy(const torch::Tensor& logits, const torch::Tensor& labels,
                int64_t ignore_index) {
  // logits: [B, S, V], labels: [B, S]
  auto preds = logits.argmax(-1);  // [B, S]
  auto mask = labels != ignore_index;
  auto correct = (preds == labels) & mask;
  auto total = mask.sum().item<double>();
  if (total == 0.0) return 0.0;
  return correct.sum().item<double>() / total;
}

double top_k_accuracy(const torch::Tensor& logits, const torch::Tensor& labels,
                      int64_t k, int64_t ignore_index) {
  // logits: [B, S, V], labels: [B, S]
  auto topk = std::get<1>(logits.topk(k, -1));  // [B, S, k]
  auto labels_expanded = labels.unsqueeze(-1).expand_as(topk);
  auto mask = labels != ignore_index;
  auto correct = (topk == labels_expanded).any(-1) & mask;  // [B, S]
  auto total = mask.sum().item<double>();
  if (total == 0.0) return 0.0;
  return correct.sum().item<double>() / total;
}

double f1_score(const torch::Tensor& predictions, const torch::Tensor& labels,
                int64_t num_classes) {
  double total_f1 = 0.0;
  int valid_classes = 0;

  for (int64_t c = 0; c < num_classes; ++c) {
    auto pred_c = predictions == c;
    auto label_c = labels == c;
    double tp = (pred_c & label_c).sum().item<double>();
    double fp = (pred_c & ~label_c).sum().item<double>();
    double fn = (~pred_c & label_c).sum().item<double>();

    if (tp + fp + fn > 0) {
      double precision = (tp + fp > 0) ? tp / (tp + fp) : 0.0;
      double recall = (tp + fn > 0) ? tp / (tp + fn) : 0.0;
      double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
      total_f1 += f1;
      valid_classes++;
    }
  }

  return (valid_classes > 0) ? total_f1 / valid_classes : 0.0;
}

}  // namespace olmo_cpp
