#pragma once

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include <torch/torch.h>
#include <optional>
#include <string>
#include <vector>
#include <memory>

namespace olmo_cpp {

class Callback;
class Evaluator;

/// Training configuration
struct TrainConfig {
  int64_t num_steps = 1000;
  int64_t batch_size = 4;
  int64_t seq_len = 128;
  double lr = 1e-4;
  int64_t warmup_steps = 100;
  int64_t grad_accum_steps = 1;
  double max_grad_norm = 1.0;
  double weight_decay = 0.01;

  // Mixed precision
  bool use_amp = false;
  bool use_grad_scaler = false;

  // LR scheduler
  std::string scheduler = "cosine";

  // Activation checkpointing
  int64_t activation_checkpoint_interval = 0;

  // Data
  std::optional<std::string> data_path;
  std::optional<std::string> eval_data_path;
  int64_t eval_interval = 500;

  // Checkpointing
  std::string checkpoint_dir;
  int64_t checkpoint_interval = 1000;
  int keep_checkpoints = 3;

  // Sequence/batch scheduling
  int64_t target_seq_len = -1;
  int64_t seq_len_warmup_steps = 0;
  int64_t target_batch_size = -1;
  int64_t batch_size_ramp_steps = 0;
};

/// Train for num_steps with AdamW
void train_epoch(
    Transformer& model,
    const TransformerConfig& cfg,
    int64_t num_steps,
    std::optional<std::string> data_path = std::nullopt,
    int64_t batch_size = 4,
    int64_t seq_len = 128,
    double lr = 1e-4,
    int64_t warmup_steps = 100,
    torch::Device device = torch::Device(torch::kCPU),
    int64_t grad_accum_steps = 1,
    bool use_amp = false);

/// Full-featured training with callbacks, schedulers, checkpointing
void train(
    Transformer& model,
    const TransformerConfig& model_cfg,
    const TrainConfig& train_cfg,
    torch::Device device,
    std::vector<std::shared_ptr<Callback>> callbacks = {},
    std::shared_ptr<Evaluator> evaluator = nullptr);

}  // namespace olmo_cpp
