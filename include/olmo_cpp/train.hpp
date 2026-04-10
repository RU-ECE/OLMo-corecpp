#pragma once

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/fused_transformer.hpp"
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
  bool use_amp = false;           // autocast: FP32 master weights, per-op BF16 (small models)
  bool use_bf16 = false;          // pure BF16: convert weights to BF16, no autocast (large models)
  bool use_grad_scaler = false;   // loss scaling for fp16

  // Optimizer selection
  std::string optimizer = "adamw";  // adamw, lion, muon, dion

  // LR scheduler
  std::string scheduler = "cosine";  // cosine, linear, constant, wsd, etc.

  // Activation checkpointing
  int64_t activation_checkpoint_interval = 0;  // 0 = disabled

  // Data
  std::optional<std::string> data_path;
  std::optional<std::string> eval_data_path;
  int64_t eval_interval = 500;  // steps between evals

  // Checkpointing
  std::string checkpoint_dir;
  int64_t checkpoint_interval = 1000;
  int keep_checkpoints = 3;

  // Sequence/batch scheduling
  int64_t target_seq_len = -1;      // for curriculum: ramp from seq_len to this
  int64_t seq_len_warmup_steps = 0;
  int64_t target_batch_size = -1;
  int64_t batch_size_ramp_steps = 0;

  // Performance optimizations
  bool use_foreach_optimizer = true;  // Use _foreach_ batched ops in AdamW
  bool gpu_resident_data = true;      // If false: pinned host + H2D streaming (no full corpus on GPU)
  /// 0 = auto VRAM budget for full GPU residency; >0 = max token count allowed on GPU;
  /// -1 = never put full corpus on GPU (streaming only).
  int64_t max_gpu_data_tokens = 0;
  int64_t log_interval = 10;         // Steps between loss D2H sync
  bool use_cuda_graph = false;       // Capture forward+backward as CUDA graph (requires fixed shapes)

  // Heartbeat monitoring
  double report_every = 300.0;       // seconds between heartbeat writes (0 = disabled)
  std::string heartbeat_path;        // file to write heartbeat to (empty = disabled)
};

/// Train for num_steps (legacy API, kept for backward compat)
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
    bool use_amp = false,
    const std::string& optimizer_name = "adamw");

/// Full-featured training with callbacks, schedulers, checkpointing
void train(
    Transformer& model,
    const TransformerConfig& model_cfg,
    const TrainConfig& train_cfg,
    torch::Device device,
    std::vector<std::shared_ptr<Callback>> callbacks = {},
    std::shared_ptr<Evaluator> evaluator = nullptr);

/// Full-featured training for FusedTransformer
void train(
    FusedTransformer& model,
    const TransformerConfig& model_cfg,
    const TrainConfig& train_cfg,
    torch::Device device,
    std::vector<std::shared_ptr<Callback>> callbacks = {},
    std::shared_ptr<Evaluator> evaluator = nullptr);

}  // namespace olmo_cpp
