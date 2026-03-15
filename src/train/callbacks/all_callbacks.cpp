#include "olmo_cpp/train/callbacks/all_callbacks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <numeric>

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace fs = std::filesystem;

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// 1. ConsoleLoggerCallback
// ---------------------------------------------------------------------------

ConsoleLoggerCallback::ConsoleLoggerCallback(int64_t log_interval, int rank)
    : log_interval_(log_interval), rank_(rank) {}

void ConsoleLoggerCallback::on_train_start(TrainState& state) {
  if (rank_ != 0) return;
  std::cout << "[ConsoleLogger] Training started." << std::endl;
  (void)state;
}

void ConsoleLoggerCallback::on_step_end(TrainState& state) {
  if (rank_ != 0) return;
  if (state.global_step % log_interval_ != 0) return;

  auto now = std::chrono::steady_clock::now();
  double step_secs = std::chrono::duration<double>(now - state.step_start).count();
  double tokens_per_sec = 0.0;
  if (step_secs > 0.0) {
    tokens_per_sec =
        static_cast<double>(state.batch_size * state.seq_len) / step_secs;
  }

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "[step " << state.global_step << "]"
            << " loss=" << state.loss
            << " lr=" << state.learning_rate
            << " grad_norm=" << state.grad_norm
            << " tok/s=" << static_cast<int64_t>(tokens_per_sec)
            << " tokens_seen=" << state.tokens_seen
            << std::endl;
}

void ConsoleLoggerCallback::on_train_end(TrainState& state) {
  if (rank_ != 0) return;
  auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - state.train_start);
  std::cout << "[ConsoleLogger] Training finished. Total steps: "
            << state.global_step
            << ", Total tokens: " << state.tokens_seen
            << ", Elapsed: " << std::fixed << std::setprecision(1)
            << elapsed.count() << "s" << std::endl;
}

// ---------------------------------------------------------------------------
// 2. SpeedMonitorCallback
// ---------------------------------------------------------------------------

SpeedMonitorCallback::SpeedMonitorCallback(int64_t window_size)
    : window_size_(window_size) {}

void SpeedMonitorCallback::on_step_start(TrainState& /*state*/) {
  last_time_ = std::chrono::steady_clock::now();
}

void SpeedMonitorCallback::on_step_end(TrainState& state) {
  auto now = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(now - last_time_).count();
  int64_t tokens_this_step = state.batch_size * state.seq_len;

  step_times_.push_back(dt);
  step_tokens_.push_back(tokens_this_step);
  if (static_cast<int64_t>(step_times_.size()) > window_size_) {
    step_times_.pop_front();
    step_tokens_.pop_front();
  }

  double total_time = std::accumulate(step_times_.begin(), step_times_.end(), 0.0);
  int64_t total_tokens = std::accumulate(step_tokens_.begin(), step_tokens_.end(), int64_t(0));

  double steps_per_sec = 0.0;
  double tokens_per_sec = 0.0;
  if (total_time > 0.0) {
    steps_per_sec = static_cast<double>(step_times_.size()) / total_time;
    tokens_per_sec = static_cast<double>(total_tokens) / total_time;
  }

  state.metrics["speed/steps_per_sec"] = static_cast<float>(steps_per_sec);
  state.metrics["speed/tokens_per_sec"] = static_cast<float>(tokens_per_sec);
  state.metrics["speed/step_time_ms"] = static_cast<float>(dt * 1000.0);
}

// ---------------------------------------------------------------------------
// 3. GPUMemoryMonitorCallback
// ---------------------------------------------------------------------------

void GPUMemoryMonitorCallback::on_step_end(TrainState& state) {
#ifdef USE_CUDA
  if (!torch::cuda::is_available()) return;

  auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);

  // allocated_bytes[0] is the stat for "all" segment pool
  int64_t allocated = stats.allocated_bytes[0].current;
  int64_t reserved = stats.reserved_bytes[0].current;
  int64_t peak_allocated = stats.allocated_bytes[0].peak;
  int64_t peak_reserved = stats.reserved_bytes[0].peak;

  double alloc_gb = static_cast<double>(allocated) / (1024.0 * 1024.0 * 1024.0);
  double reserved_gb = static_cast<double>(reserved) / (1024.0 * 1024.0 * 1024.0);
  double peak_alloc_gb = static_cast<double>(peak_allocated) / (1024.0 * 1024.0 * 1024.0);
  double peak_reserved_gb = static_cast<double>(peak_reserved) / (1024.0 * 1024.0 * 1024.0);

  state.metrics["gpu/allocated_gb"] = static_cast<float>(alloc_gb);
  state.metrics["gpu/reserved_gb"] = static_cast<float>(reserved_gb);
  state.metrics["gpu/peak_allocated_gb"] = static_cast<float>(peak_alloc_gb);
  state.metrics["gpu/peak_reserved_gb"] = static_cast<float>(peak_reserved_gb);
#else
  (void)state;
#endif
}

// ---------------------------------------------------------------------------
// 4. StabilityMonitorCallback
// ---------------------------------------------------------------------------

StabilityMonitorCallback::StabilityMonitorCallback(double spike_threshold,
                                                   int64_t window_size)
    : spike_threshold_(spike_threshold), window_size_(window_size) {}

void StabilityMonitorCallback::on_step_end(TrainState& state) {
  float loss = state.loss;

  // Check for NaN / Inf
  if (std::isnan(loss) || std::isinf(loss)) {
    nan_count_++;
    std::cerr << "[StabilityMonitor] WARNING: NaN/Inf loss detected at step "
              << state.global_step << " (total NaN/Inf count: " << nan_count_
              << ")" << std::endl;
    state.metrics["stability/nan_count"] = static_cast<float>(nan_count_);
    return;
  }

  loss_history_.push_back(loss);
  if (static_cast<int64_t>(loss_history_.size()) > window_size_) {
    loss_history_.pop_front();
  }

  // Need at least a few samples to compute stats
  if (loss_history_.size() < 5) return;

  double sum = std::accumulate(loss_history_.begin(), loss_history_.end(), 0.0);
  double mean = sum / static_cast<double>(loss_history_.size());

  double sq_sum = 0.0;
  for (float l : loss_history_) {
    double diff = static_cast<double>(l) - mean;
    sq_sum += diff * diff;
  }
  double stddev = std::sqrt(sq_sum / static_cast<double>(loss_history_.size()));

  double threshold = mean + spike_threshold_ * stddev;
  if (static_cast<double>(loss) > threshold && stddev > 1e-6) {
    spike_count_++;
    std::cerr << "[StabilityMonitor] WARNING: Loss spike detected at step "
              << state.global_step << ": loss=" << loss
              << " (mean=" << mean << ", std=" << stddev
              << ", threshold=" << threshold
              << ", total spikes: " << spike_count_ << ")" << std::endl;
  }

  state.metrics["stability/loss_mean"] = static_cast<float>(mean);
  state.metrics["stability/loss_std"] = static_cast<float>(stddev);
  state.metrics["stability/spike_count"] = static_cast<float>(spike_count_);
  state.metrics["stability/nan_count"] = static_cast<float>(nan_count_);
}

// ---------------------------------------------------------------------------
// 5. CheckpointerCallback
// ---------------------------------------------------------------------------

CheckpointerCallback::CheckpointerCallback(const std::string& save_dir,
                                           int64_t save_interval,
                                           int64_t keep_last_n, int rank)
    : save_dir_(save_dir),
      save_interval_(save_interval),
      keep_last_n_(keep_last_n),
      rank_(rank) {}

void CheckpointerCallback::on_step_end(TrainState& state) {
  if (state.global_step % save_interval_ != 0) return;
  if (state.global_step == 0) return;
  save_checkpoint(state);
}

void CheckpointerCallback::on_train_end(TrainState& state) {
  save_checkpoint(state);
}

void CheckpointerCallback::save_checkpoint(TrainState& state) {
  if (rank_ != 0) return;

  fs::create_directories(save_dir_);

  std::string ckpt_name = "step_" + std::to_string(state.global_step);
  std::string ckpt_dir = save_dir_ + "/" + ckpt_name;
  fs::create_directories(ckpt_dir);

  // Save model weights if available
  if (model_) {
    std::string model_path = ckpt_dir + "/model.pt";
    torch::serialize::OutputArchive archive;
    model_->save(archive);
    archive.save_to(model_path);
    std::cout << "[Checkpointer] Saved model to " << model_path << std::endl;
  }

  // Save training state metadata
  std::string meta_path = ckpt_dir + "/train_state.json";
  std::ofstream meta(meta_path);
  if (meta.is_open()) {
    meta << "{\n"
         << "  \"global_step\": " << state.global_step << ",\n"
         << "  \"epoch\": " << state.epoch << ",\n"
         << "  \"tokens_seen\": " << state.tokens_seen << ",\n"
         << "  \"loss\": " << state.loss << ",\n"
         << "  \"learning_rate\": " << state.learning_rate << "\n"
         << "}" << std::endl;
    meta.close();
  }

  saved_paths_.push_back(ckpt_dir);

  // Remove old checkpoints beyond keep_last_n
  while (keep_last_n_ > 0 &&
         static_cast<int64_t>(saved_paths_.size()) > keep_last_n_) {
    std::string old_path = saved_paths_.front();
    saved_paths_.pop_front();
    std::error_code ec;
    fs::remove_all(old_path, ec);
    if (!ec) {
      std::cout << "[Checkpointer] Removed old checkpoint: " << old_path
                << std::endl;
    }
  }
}

// ---------------------------------------------------------------------------
// 6. GradientMonitorCallback
// ---------------------------------------------------------------------------

GradientMonitorCallback::GradientMonitorCallback(int64_t log_interval)
    : log_interval_(log_interval) {}

void GradientMonitorCallback::on_after_backward(TrainState& state) {
  if (state.global_step % log_interval_ != 0) return;
  if (!model_) return;

  float min_norm = std::numeric_limits<float>::max();
  float max_norm = 0.0f;
  float total_norm = 0.0f;
  int64_t param_count = 0;

  for (const auto& pair : model_->named_parameters()) {
    const auto& param = pair.value();
    if (!param.grad().defined()) continue;

    float norm = param.grad().norm().item<float>();
    min_norm = std::min(min_norm, norm);
    max_norm = std::max(max_norm, norm);
    total_norm += norm;
    param_count++;
  }

  if (param_count > 0) {
    float mean_norm = total_norm / static_cast<float>(param_count);
    state.metrics["grad/min_norm"] = min_norm;
    state.metrics["grad/max_norm"] = max_norm;
    state.metrics["grad/mean_norm"] = mean_norm;
    state.metrics["grad/num_params"] = static_cast<float>(param_count);

    std::cout << "[GradientMonitor] step=" << state.global_step
              << " grad_norms: min=" << min_norm << " max=" << max_norm
              << " mean=" << mean_norm << " params=" << param_count
              << std::endl;
  }
}

// ---------------------------------------------------------------------------
// 7. MetricSaverCallback
// ---------------------------------------------------------------------------

MetricSaverCallback::MetricSaverCallback(const std::string& output_path,
                                         int64_t save_interval)
    : output_path_(output_path), save_interval_(save_interval) {}

void MetricSaverCallback::on_step_end(TrainState& state) {
  // Copy current metrics plus standard fields into the buffer
  std::unordered_map<std::string, float> entry = state.metrics;
  entry["global_step"] = static_cast<float>(state.global_step);
  entry["epoch"] = static_cast<float>(state.epoch);
  entry["loss"] = state.loss;
  entry["learning_rate"] = state.learning_rate;
  entry["grad_norm"] = state.grad_norm;
  entry["tokens_seen"] = static_cast<float>(state.tokens_seen);
  buffer_.push_back(std::move(entry));

  if (static_cast<int64_t>(buffer_.size()) >= save_interval_) {
    flush();
  }
}

void MetricSaverCallback::on_train_end(TrainState& /*state*/) {
  if (!buffer_.empty()) {
    flush();
  }
}

void MetricSaverCallback::flush() {
  // Ensure parent directory exists
  fs::path p(output_path_);
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path());
  }

  // Append to file as a JSON array (one array per flush)
  std::ofstream out(output_path_, std::ios::app);
  if (!out.is_open()) {
    std::cerr << "[MetricSaver] Failed to open " << output_path_ << std::endl;
    buffer_.clear();
    return;
  }

  out << "[\n";
  for (size_t i = 0; i < buffer_.size(); ++i) {
    out << "  {";
    bool first = true;
    for (const auto& kv : buffer_[i]) {
      if (!first) out << ", ";
      out << "\"" << kv.first << "\": " << kv.second;
      first = false;
    }
    out << "}";
    if (i + 1 < buffer_.size()) out << ",";
    out << "\n";
  }
  out << "]\n";
  out.close();

  buffer_.clear();
}

// ---------------------------------------------------------------------------
// 8. ProfilerCallback
// ---------------------------------------------------------------------------

ProfilerCallback::ProfilerCallback(int64_t start_step, int64_t num_steps,
                                   const std::string& trace_dir)
    : start_step_(start_step), num_steps_(num_steps), trace_dir_(trace_dir) {}

void ProfilerCallback::on_step_start(TrainState& state) {
  if (state.global_step == start_step_) {
    active_ = true;
    fs::create_directories(trace_dir_);
    std::cout << "[Profiler] Profiling started at step " << state.global_step
              << std::endl;
  }
}

void ProfilerCallback::on_step_end(TrainState& state) {
  if (!active_) return;

  auto now = std::chrono::steady_clock::now();
  double step_ms =
      std::chrono::duration<double, std::milli>(now - state.step_start).count();

  // Write a simple trace event in Chrome Trace Format
  std::string trace_file =
      trace_dir_ + "/step_" + std::to_string(state.global_step) + ".json";
  std::ofstream out(trace_file);
  if (out.is_open()) {
    auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       now.time_since_epoch())
                       .count();
    out << "{\n"
        << "  \"traceEvents\": [\n"
        << "    {\n"
        << "      \"name\": \"training_step\",\n"
        << "      \"cat\": \"train\",\n"
        << "      \"ph\": \"X\",\n"
        << "      \"ts\": " << (wall_us - static_cast<int64_t>(step_ms * 1000.0))
        << ",\n"
        << "      \"dur\": " << static_cast<int64_t>(step_ms * 1000.0) << ",\n"
        << "      \"pid\": 0,\n"
        << "      \"tid\": 0,\n"
        << "      \"args\": {\n"
        << "        \"step\": " << state.global_step << ",\n"
        << "        \"loss\": " << state.loss << ",\n"
        << "        \"step_time_ms\": " << step_ms << "\n"
        << "      }\n"
        << "    }\n"
        << "  ]\n"
        << "}" << std::endl;
    out.close();
  }

  if (state.global_step >= start_step_ + num_steps_ - 1) {
    active_ = false;
    std::cout << "[Profiler] Profiling finished. Traces written to "
              << trace_dir_ << std::endl;
  }
}

// ---------------------------------------------------------------------------
// 9. GarbageCollectorCallback
// ---------------------------------------------------------------------------

GarbageCollectorCallback::GarbageCollectorCallback(int64_t interval)
    : interval_(interval) {}

void GarbageCollectorCallback::on_step_end(TrainState& state) {
  if (state.global_step % interval_ != 0) return;
  if (state.global_step == 0) return;

#ifdef USE_CUDA
  if (torch::cuda::is_available()) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
#endif
}

// ---------------------------------------------------------------------------
// 10. ConfigSaverCallback
// ---------------------------------------------------------------------------

ConfigSaverCallback::ConfigSaverCallback(const std::string& save_path)
    : save_path_(save_path) {}

void ConfigSaverCallback::on_train_start(TrainState& state) {
  fs::path p(save_path_);
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path());
  }

  std::ofstream out(save_path_);
  if (!out.is_open()) {
    std::cerr << "[ConfigSaver] Failed to open " << save_path_ << std::endl;
    return;
  }

  out << "{\n"
      << "  \"batch_size\": " << state.batch_size << ",\n"
      << "  \"seq_len\": " << state.seq_len << ",\n"
      << "  \"learning_rate\": " << state.learning_rate << ",\n"
      << "  \"epoch\": " << state.epoch << ",\n"
      << "  \"global_step\": " << state.global_step << ",\n"
      << "  \"tokens_seen\": " << state.tokens_seen << "\n"
      << "}" << std::endl;
  out.close();

  std::cout << "[ConfigSaver] Training config saved to " << save_path_
            << std::endl;
}

// ---------------------------------------------------------------------------
// 11. SequenceLengthSchedulerCallback
// ---------------------------------------------------------------------------

SequenceLengthSchedulerCallback::SequenceLengthSchedulerCallback(
    int64_t initial_seq_len, int64_t target_seq_len, int64_t warmup_steps)
    : initial_seq_len_(initial_seq_len),
      target_seq_len_(target_seq_len),
      warmup_steps_(warmup_steps),
      current_seq_len_(initial_seq_len) {}

void SequenceLengthSchedulerCallback::on_step_start(TrainState& state) {
  if (warmup_steps_ <= 0) {
    current_seq_len_ = target_seq_len_;
  } else if (state.global_step >= warmup_steps_) {
    current_seq_len_ = target_seq_len_;
  } else {
    // Linear interpolation
    double frac =
        static_cast<double>(state.global_step) / static_cast<double>(warmup_steps_);
    current_seq_len_ = initial_seq_len_ +
        static_cast<int64_t>(frac * static_cast<double>(target_seq_len_ - initial_seq_len_));
    // Round to nearest multiple of 64 for efficiency
    current_seq_len_ = ((current_seq_len_ + 63) / 64) * 64;
    // Clamp
    current_seq_len_ = std::min(current_seq_len_, target_seq_len_);
    current_seq_len_ = std::max(current_seq_len_, initial_seq_len_);
  }
  state.seq_len = current_seq_len_;
}

// ---------------------------------------------------------------------------
// 12. BatchSizeSchedulerCallback
// ---------------------------------------------------------------------------

BatchSizeSchedulerCallback::BatchSizeSchedulerCallback(
    int64_t initial_batch_size, int64_t target_batch_size,
    int64_t warmup_steps)
    : initial_batch_size_(initial_batch_size),
      target_batch_size_(target_batch_size),
      warmup_steps_(warmup_steps),
      current_batch_size_(initial_batch_size) {}

void BatchSizeSchedulerCallback::on_step_start(TrainState& state) {
  if (warmup_steps_ <= 0 || state.global_step >= warmup_steps_) {
    current_batch_size_ = target_batch_size_;
  } else {
    // Double the batch size at evenly spaced intervals until reaching target.
    // Compute how many doublings we need.
    int64_t bs = initial_batch_size_;
    int num_doublings = 0;
    int64_t tmp = bs;
    while (tmp < target_batch_size_) {
      tmp *= 2;
      num_doublings++;
    }

    if (num_doublings == 0) {
      current_batch_size_ = target_batch_size_;
    } else {
      int64_t steps_per_doubling = warmup_steps_ / num_doublings;
      if (steps_per_doubling <= 0) steps_per_doubling = 1;
      int doublings_done = static_cast<int>(state.global_step / steps_per_doubling);
      doublings_done = std::min(doublings_done, num_doublings);
      current_batch_size_ = initial_batch_size_ * (int64_t(1) << doublings_done);
      current_batch_size_ = std::min(current_batch_size_, target_batch_size_);
    }
  }
  state.batch_size = current_batch_size_;
}

// ---------------------------------------------------------------------------
// 13. EvaluatorCallback
// ---------------------------------------------------------------------------

EvaluatorCallback::EvaluatorCallback(EvalFn eval_fn, int64_t eval_interval)
    : eval_fn_(std::move(eval_fn)), eval_interval_(eval_interval) {}

void EvaluatorCallback::on_step_end(TrainState& state) {
  if (state.global_step % eval_interval_ != 0) return;
  if (state.global_step == 0) return;

  auto eval_metrics = eval_fn_();

  // Merge eval metrics into state.metrics with "eval/" prefix
  for (const auto& kv : eval_metrics) {
    state.metrics["eval/" + kv.first] = kv.second;
  }

  // Note: the CallbackManager should call on_eval_end after this, but since
  // EvaluatorCallback itself is called via on_step_end, higher-level code
  // should handle the on_eval_end dispatch. We store metrics so they are
  // available.
}

// ---------------------------------------------------------------------------
// 14. WandBCallback
// ---------------------------------------------------------------------------

WandBCallback::WandBCallback(const std::string& project,
                             const std::string& run_name,
                             const std::string& entity)
    : project_(project), run_name_(run_name), entity_(entity) {}

void WandBCallback::on_train_start(TrainState& state) {
  (void)state;
  // Create a local log directory for JSONL logging
  log_dir_ = "./wandb_logs";
  fs::create_directories(log_dir_);

  if (run_name_.empty()) {
    run_name_ = "run_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
  }

  // Write run metadata
  std::string meta_path = log_dir_ + "/" + run_name_ + "_meta.json";
  std::ofstream meta(meta_path);
  if (meta.is_open()) {
    meta << "{\n"
         << "  \"project\": \"" << project_ << "\",\n"
         << "  \"run_name\": \"" << run_name_ << "\",\n"
         << "  \"entity\": \"" << entity_ << "\"\n"
         << "}" << std::endl;
    meta.close();
  }

  initialized_ = true;
  std::cout << "[WandB] Initialized run '" << run_name_ << "' in project '"
            << project_ << "'. Logs at " << log_dir_ << std::endl;
}

void WandBCallback::on_step_end(TrainState& state) {
  if (!initialized_) return;

  // Append metrics as JSONL (one JSON object per line)
  std::string log_path = log_dir_ + "/" + run_name_ + "_metrics.jsonl";
  std::ofstream out(log_path, std::ios::app);
  if (!out.is_open()) return;

  out << "{\"step\": " << state.global_step
      << ", \"loss\": " << state.loss
      << ", \"lr\": " << state.learning_rate
      << ", \"grad_norm\": " << state.grad_norm
      << ", \"tokens_seen\": " << state.tokens_seen;

  for (const auto& kv : state.metrics) {
    out << ", \"" << kv.first << "\": " << kv.second;
  }
  out << "}\n";
  out.close();
}

void WandBCallback::on_train_end(TrainState& state) {
  if (!initialized_) return;

  // Write summary
  std::string summary_path = log_dir_ + "/" + run_name_ + "_summary.json";
  std::ofstream out(summary_path);
  if (out.is_open()) {
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - state.train_start);
    out << "{\n"
        << "  \"final_step\": " << state.global_step << ",\n"
        << "  \"final_loss\": " << state.loss << ",\n"
        << "  \"total_tokens\": " << state.tokens_seen << ",\n"
        << "  \"elapsed_seconds\": " << elapsed.count() << "\n"
        << "}" << std::endl;
    out.close();
  }

  std::cout << "[WandB] Run '" << run_name_ << "' finished. Summary at "
            << summary_path << std::endl;
}

// ---------------------------------------------------------------------------
// 15. SlackNotifierCallback
// ---------------------------------------------------------------------------

SlackNotifierCallback::SlackNotifierCallback(const std::string& webhook_url,
                                             int64_t notify_interval)
    : webhook_url_(webhook_url), notify_interval_(notify_interval) {}

void SlackNotifierCallback::on_train_start(TrainState& /*state*/) {
  send_message("Training started.");
}

void SlackNotifierCallback::on_train_end(TrainState& state) {
  std::ostringstream oss;
  oss << "Training finished. Final step: " << state.global_step
      << ", Final loss: " << state.loss
      << ", Total tokens: " << state.tokens_seen;
  send_message(oss.str());
}

void SlackNotifierCallback::on_step_end(TrainState& state) {
  if (state.global_step % notify_interval_ != 0) return;
  if (state.global_step == 0) return;

  std::ostringstream oss;
  oss << "Step " << state.global_step
      << " | loss=" << std::fixed << std::setprecision(4) << state.loss
      << " | lr=" << state.learning_rate
      << " | tokens=" << state.tokens_seen;
  send_message(oss.str());
}

void SlackNotifierCallback::send_message(const std::string& text) {
  // Escape double quotes in text for JSON
  std::string escaped;
  escaped.reserve(text.size());
  for (char c : text) {
    if (c == '"') escaped += "\\\"";
    else if (c == '\\') escaped += "\\\\";
    else escaped += c;
  }

  std::string cmd = "curl -s -X POST -H 'Content-type: application/json' "
                    "--data '{\"text\": \"" +
                    escaped + "\"}' '" + webhook_url_ + "' > /dev/null 2>&1 &";
  // Run asynchronously to avoid blocking training
  std::system(cmd.c_str());
}

// ---------------------------------------------------------------------------
// 16. ModelMergerCallback (EMA)
// ---------------------------------------------------------------------------

ModelMergerCallback::ModelMergerCallback(double ema_decay,
                                         int64_t update_interval)
    : ema_decay_(ema_decay), update_interval_(update_interval) {}

void ModelMergerCallback::on_step_end(TrainState& state) {
  if (!model_) return;
  if (state.global_step % update_interval_ != 0) return;

  auto params = model_->parameters();

  if (!initialized_) {
    // Initialize EMA params as clones of model params
    ema_params_.clear();
    ema_params_.reserve(params.size());
    for (const auto& p : params) {
      ema_params_.push_back(p.detach().clone());
    }
    initialized_ = true;
    return;
  }

  // Update: ema = decay * ema + (1 - decay) * param
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < params.size(); ++i) {
    ema_params_[i].mul_(ema_decay_).add_(params[i].detach(), 1.0 - ema_decay_);
  }
}

}  // namespace olmo_cpp
