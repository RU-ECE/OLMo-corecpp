// Full-featured training loop with callbacks, multiple optimizers,
// LR schedulers, activation checkpointing, gradient scaling, and eval
#include "olmo_cpp/train.hpp"
#include "olmo_cpp/data/token_dataset.hpp"
#include "olmo_cpp/profiler.hpp"
#include <ATen/autocast_mode.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include "olmo_cpp/distributed/ddp.hpp"
#include "olmo_cpp/optim/lion.hpp"
#include "olmo_cpp/optim/muon.hpp"
#include "olmo_cpp/optim/dion.hpp"
#include "olmo_cpp/optim/foreach_adamw.hpp"
#include "olmo_cpp/optim/grad_clip.hpp"
#include "olmo_cpp/optim/skip_step.hpp"
#include "olmo_cpp/optim/scheduler.hpp"
#include "olmo_cpp/train/callback.hpp"
#include "olmo_cpp/train/grad_scaler.hpp"
#include "olmo_cpp/train/activation_checkpoint.hpp"
#include "olmo_cpp/train/checkpoint.hpp"
#include "olmo_cpp/eval/evaluator.hpp"
#include "olmo_cpp/io/filesystem.hpp"
#include <torch/nn/init.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// RAII autocast guard — uses the non-deprecated PyTorch 2.6+ API
// ---------------------------------------------------------------------------
namespace {

struct AutocastGuard {
  explicit AutocastGuard(bool enabled, torch::Device device) : enabled_(enabled && device.is_cuda()) {
    if (enabled_) {
      prev_enabled_ = at::autocast::is_autocast_enabled(at::kCUDA);
      prev_dtype_ = at::autocast::get_autocast_dtype(at::kCUDA);
      at::autocast::set_autocast_enabled(at::kCUDA, true);
      at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
      at::autocast::increment_nesting();
    }
  }
  ~AutocastGuard() {
    if (enabled_) {
      at::autocast::decrement_nesting();
      at::autocast::clear_cache();
      at::autocast::set_autocast_enabled(at::kCUDA, prev_enabled_);
      at::autocast::set_autocast_dtype(at::kCUDA, prev_dtype_);
    }
  }
  AutocastGuard(const AutocastGuard&) = delete;
  AutocastGuard& operator=(const AutocastGuard&) = delete;
 private:
  bool enabled_;
  bool prev_enabled_{false};
  at::ScalarType prev_dtype_{at::kFloat};
};

double cosine_warmup_lr(int64_t step, int64_t warmup_steps, double base_lr, int64_t total_steps) {
  if (step < warmup_steps) {
    return base_lr * static_cast<double>(step + 1) / static_cast<double>(warmup_steps);
  }
  double progress = static_cast<double>(step - warmup_steps) / static_cast<double>(total_steps - warmup_steps);
  return 0.5 * base_lr * (1.0 + std::cos(M_PI * progress));
}

std::string optimizer_display_name(const std::string& name, bool use_foreach) {
  if (name == "adamw" && use_foreach) return "ForeachAdamW (batched _foreach_* ops)";
  if (name == "adamw") return "AdamW (standard per-param)";
  return name;
}

void print_optimization_banner(const std::string& optimizer_name, bool use_foreach,
                               bool gpu_data, bool gpu_data_active,
                               bool use_amp, bool fused_grad_clip,
                               bool cuda_graph = false) {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║  OPTIMIZATION STATUS                                    ║\n";
  std::cout << "╠══════════════════════════════════════════════════════════╣\n";
  std::cout << "║  Optimizer:       " << std::left << std::setw(39)
            << optimizer_display_name(optimizer_name, use_foreach) << "║\n";
  std::cout << "║  Grad clipping:   " << std::left << std::setw(39)
            << "GPU-resident (foreach_norm, no D2H)" << "║\n";
  std::cout << "║  Data loading:    " << std::left << std::setw(39)
            << (gpu_data_active ? "GPU-resident (zero per-step H2D)" : "CPU + async prefetch") << "║\n";
  std::cout << "║  Mixed precision: " << std::left << std::setw(39)
            << (use_amp ? "BF16 autocast" : "FP32") << "║\n";
  std::cout << "║  CUDA graph:      " << std::left << std::setw(39)
            << (cuda_graph ? "ON (fwd+bwd captured)" : "OFF") << "║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";
  std::cout << "\n";
}

}  // namespace

// ---------------------------------------------------------------------------
// Legacy train_epoch (backward compatible)
// ---------------------------------------------------------------------------

void train_epoch(
    Transformer& model,
    const TransformerConfig& cfg,
    int64_t num_steps,
    std::optional<std::string> data_path,
    int64_t batch_size,
    int64_t seq_len,
    double lr,
    int64_t warmup_steps,
    torch::Device device,
    int64_t grad_accum_steps,
    bool use_amp,
    const std::string& optimizer_name) {
  model->train();

  // Create optimizer based on selection
  std::unique_ptr<torch::optim::Optimizer> opt;
  if (optimizer_name == "muon") {
    opt = std::make_unique<Muon>(model->parameters(), MuonOptions(lr));
  } else if (optimizer_name == "lion") {
    opt = std::make_unique<Lion>(model->parameters(), LionOptions(lr).weight_decay(0.01));
  } else if (optimizer_name == "dion") {
    opt = std::make_unique<DION>(model->parameters(), DIONOptions(lr).weight_decay(0.01));
  } else {
    opt = std::make_unique<ForeachAdamW>(
        model->parameters(), ForeachAdamWOptions(lr).weight_decay(0.01));
  }
  auto& optimizer = *opt;

  auto ddp = DDPContext::init_from_env();
  if (ddp && ddp->is_distributed()) {
    auto params = model->parameters();
    ddp->broadcast_parameters(params);
  }

  std::optional<TokenDataset> dataset;
  bool gpu_data_active = false;
  if (data_path && !data_path->empty()) {
    dataset.emplace(*data_path, seq_len, true);
    dataset->reset_epoch();
    dataset->to_device(device, 0);
    gpu_data_active = dataset->is_gpu_resident();
  }

  if (!ddp || ddp->rank() == 0) {
    print_optimization_banner(optimizer_name, true, true, gpu_data_active, use_amp, true);
  }

  // Pre-build DDP parameter list once (avoids per-step allocation)
  std::vector<torch::Tensor> ddp_params;
  if (ddp && ddp->is_distributed()) {
    for (auto& p : model->parameters()) ddp_params.push_back(p);
  }

  // Epoch tracking
  int64_t tokens_per_step = batch_size * seq_len * grad_accum_steps;
  int64_t dataset_tokens = dataset ? dataset->size() * seq_len : 0;
  int64_t steps_per_epoch = (dataset && dataset_tokens > 0)
      ? std::max(int64_t(1), dataset_tokens / tokens_per_step) : num_steps;
  int64_t current_epoch = 0;

  auto train_start = std::chrono::steady_clock::now();
  int64_t total_tokens = 0;
  double epoch_loss_sum = 0.0;
  int64_t epoch_loss_count = 0;

  // Pre-allocate loss accumulator (avoids per-step allocation)
  torch::Tensor accum_loss_tensor = torch::zeros({}, torch::TensorOptions().device(device));

  // Cache model params once (avoids per-step vector allocation)
  auto model_params = model->parameters();

  for (int64_t step = 0; step < num_steps; ++step) {
    auto step_start = std::chrono::steady_clock::now();
    double step_lr = cosine_warmup_lr(step, warmup_steps, lr, num_steps);
    // Set LR on the optimizer's param group
    if (optimizer_name == "muon") {
      static_cast<MuonOptions&>(optimizer.param_groups()[0].options()).lr(step_lr);
    } else if (optimizer_name == "lion") {
      static_cast<LionOptions&>(optimizer.param_groups()[0].options()).lr(step_lr);
    } else if (optimizer_name == "dion") {
      static_cast<DIONOptions&>(optimizer.param_groups()[0].options()).lr(step_lr);
    } else {
      static_cast<ForeachAdamWOptions&>(optimizer.param_groups()[0].options()).lr(step_lr);
    }

    // Epoch boundary detection
    int64_t new_epoch = dataset ? (step / steps_per_epoch) : 0;
    if (new_epoch > current_epoch && (!ddp || ddp->rank() == 0)) {
      double avg_epoch_loss = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : 0.0;
      std::cout << "--- Epoch " << current_epoch << " complete | avg_loss: "
                << std::fixed << std::setprecision(4) << avg_epoch_loss << " ---" << std::endl;
      epoch_loss_sum = 0.0;
      epoch_loss_count = 0;
      current_epoch = new_epoch;
    }

    {
      ProfileScope step_scope("step_total");
      // set_to_none=true: use nullptr instead of memset (faster)
      optimizer.zero_grad(true);
      accum_loss_tensor.zero_();

      for (int64_t accum = 0; accum < grad_accum_steps; ++accum) {
        torch::Tensor input, labels;
        {
          ProfileScope data_scope("data_loading");
          if (dataset) {
            auto [in, lab] = dataset->get_batch(batch_size, device);
            input = in; labels = lab;
            dataset->prefetch_next(batch_size, device);
          } else {
            input = torch::randint(0, cfg.vocab_size, {batch_size, seq_len},
                                   torch::TensorOptions().dtype(torch::kLong).device(device));
            labels = torch::randint(0, cfg.vocab_size, {batch_size, seq_len},
                                    torch::TensorOptions().dtype(torch::kLong).device(device));
          }
        }

        torch::Tensor loss;
        {
          ProfileScope fwd_scope("forward");
          AutocastGuard ac(use_amp, device);
          loss = model->forward(input, labels, -100) / static_cast<float>(grad_accum_steps);
        }
        {
          ProfileScope bwd_scope("backward");
          loss.backward();
        }
        accum_loss_tensor.add_(loss.detach());
      }

      {
        ProfileScope allreduce_scope("allreduce");
        if (ddp && ddp->is_distributed()) {
          ddp->allreduce_gradients(ddp_params);
        }
      }

      {
        ProfileScope optim_scope("optimizer_step");
        clip_grad_norm_gpu(model_params, 1.0);
        optimizer.step();
      }

      total_tokens += tokens_per_step;

      // Defer D2H sync: only pull loss from GPU on log steps
      if (step % 10 == 0 && (!ddp || ddp->rank() == 0)) {
        float accum_loss = accum_loss_tensor.item<float>();
        epoch_loss_sum += accum_loss;
        epoch_loss_count++;
        auto step_end = std::chrono::steady_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
        double elapsed_s = std::chrono::duration<double>(step_end - train_start).count();
        double tok_per_s = total_tokens / (elapsed_s > 0 ? elapsed_s : 1);
        double eta_s = (num_steps - step) * (elapsed_s / (step > 0 ? step : 1));

        std::cout << "Epoch " << current_epoch
                  << " | Step " << step << "/" << num_steps
                  << "  loss: " << std::fixed << std::setprecision(4) << accum_loss
                  << "  lr: " << std::scientific << std::setprecision(2) << step_lr
                  << "  step_ms: " << static_cast<int>(step_ms)
                  << "  tok/s: " << static_cast<int>(tok_per_s)
                  << "  ETA: " << static_cast<int>(eta_s / 60) << "m"
                  << std::endl;
      }
    }  // end step_scope
  }

  if (!ddp || ddp->rank() == 0) {
    auto end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(end - train_start).count();
    double avg_step_ms = total_s / num_steps * 1000.0;
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "  Steps: " << num_steps << " (" << current_epoch + 1 << " epochs)" << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Wall time: " << std::fixed << std::setprecision(2) << total_s << "s" << std::endl;
    std::cout << "  Avg step: " << std::fixed << std::setprecision(1) << avg_step_ms << "ms" << std::endl;
    std::cout << "  Throughput: " << static_cast<int>(total_tokens / total_s) << " tok/s" << std::endl;
    std::cout << "========================" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Full-featured train() with all infrastructure
// ---------------------------------------------------------------------------

void train(
    Transformer& model,
    const TransformerConfig& model_cfg,
    const TrainConfig& cfg,
    torch::Device device,
    std::vector<std::shared_ptr<Callback>> callbacks,
    std::shared_ptr<Evaluator> evaluator) {

  model->train();

  // ---- Initialize DDP ----
  auto ddp = DDPContext::init_from_env();
  if (ddp && ddp->is_distributed()) {
    auto params = model->parameters();
    ddp->broadcast_parameters(params);
  }
  int rank = ddp ? ddp->rank() : 0;

  // ---- Create optimizer ----
  std::unique_ptr<torch::optim::Optimizer> optimizer;
  std::string optim_display;
  if (cfg.optimizer == "lion") {
    optimizer = std::make_unique<Lion>(
        model->parameters(), LionOptions(cfg.lr).weight_decay(cfg.weight_decay));
    optim_display = "Lion";
  } else if (cfg.optimizer == "muon") {
    optimizer = std::make_unique<Muon>(
        model->parameters(), MuonOptions(cfg.lr));
    optim_display = "Muon";
  } else if (cfg.optimizer == "dion") {
    optimizer = std::make_unique<DION>(
        model->parameters(), DIONOptions(cfg.lr).weight_decay(cfg.weight_decay));
    optim_display = "DION";
  } else if (cfg.use_foreach_optimizer) {
    optimizer = std::make_unique<ForeachAdamW>(
        model->parameters(), ForeachAdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));
    optim_display = "ForeachAdamW";
  } else {
    optimizer = std::make_unique<torch::optim::AdamW>(
        model->parameters(),
        torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));
    optim_display = "AdamW (standard)";
  }

  // ---- Create LR scheduler ----
  auto scheduler = create_scheduler(cfg.scheduler, cfg.lr, cfg.warmup_steps);

  // ---- Gradient scaler for mixed precision ----
  std::optional<GradScaler> grad_scaler;
  if (cfg.use_grad_scaler) {
    grad_scaler.emplace();
  }

  // ---- Checkpoint manager ----
  std::optional<CheckpointManager> ckpt_mgr;
  if (!cfg.checkpoint_dir.empty()) {
    ckpt_mgr.emplace(cfg.checkpoint_dir);
  }

  // ---- Dataset ----
  std::optional<TokenDataset> dataset;
  bool gpu_data_active = false;
  if (cfg.data_path && !cfg.data_path->empty()) {
    dataset.emplace(*cfg.data_path, cfg.seq_len, true);
    dataset->reset_epoch();
    if (device.is_cuda()) {
      const int64_t cap = cfg.gpu_resident_data ? cfg.max_gpu_data_tokens : -1;
      dataset->to_device(device, cap);
      gpu_data_active = dataset->is_gpu_resident();
    } else if (cfg.gpu_resident_data) {
      dataset->to_device(device, 0);
      gpu_data_active = dataset->is_gpu_resident();
    }
  }

  if (rank == 0) {
    print_optimization_banner(cfg.optimizer, cfg.use_foreach_optimizer,
                              cfg.gpu_resident_data, gpu_data_active,
                              cfg.use_amp, true, cfg.use_cuda_graph);
  }

  // Pre-build DDP parameter list once (avoids per-step allocation)
  std::vector<torch::Tensor> ddp_params;
  if (ddp && ddp->is_distributed()) {
    for (auto& p : model->parameters()) ddp_params.push_back(p);
  }

  // ---- Callback manager ----
  CallbackManager cb_mgr;
  for (auto& cb : callbacks) cb_mgr.add(cb);

  // ---- Epoch tracking ----
  int64_t tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps;
  int64_t dataset_tokens = dataset ? dataset->size() * cfg.seq_len : 0;
  int64_t steps_per_epoch = (dataset && dataset_tokens > 0)
      ? std::max(int64_t(1), dataset_tokens / tokens_per_step) : cfg.num_steps;
  int64_t current_epoch = 0;
  double epoch_loss_sum = 0.0;
  int64_t epoch_loss_count = 0;

  // ---- TrainState ----
  TrainState state;
  state.train_start = std::chrono::steady_clock::now();
  cb_mgr.on_train_start(state);

  int64_t total_tokens = 0;

  // Pre-allocate loss accumulator and cache params (avoid per-step allocation)
  torch::Tensor accum_loss_tensor = torch::zeros({}, torch::TensorOptions().device(device));
  auto model_params = model->parameters();

  // ---- Training loop ----
  for (int64_t step = 0; step < cfg.num_steps; ++step) {
    state.step_start = std::chrono::steady_clock::now();
    state.global_step = step;

    // Sequence length scheduling (curriculum)
    int64_t cur_seq_len = cfg.seq_len;
    if (cfg.target_seq_len > 0 && cfg.seq_len_warmup_steps > 0 && step < cfg.seq_len_warmup_steps) {
      double frac = static_cast<double>(step) / cfg.seq_len_warmup_steps;
      cur_seq_len = cfg.seq_len + static_cast<int64_t>(frac * (cfg.target_seq_len - cfg.seq_len));
      cur_seq_len = ((cur_seq_len + 63) / 64) * 64;
    } else if (cfg.target_seq_len > 0) {
      cur_seq_len = cfg.target_seq_len;
    }
    state.seq_len = cur_seq_len;

    // Batch size scheduling
    int64_t cur_batch_size = cfg.batch_size;
    if (cfg.target_batch_size > 0 && cfg.batch_size_ramp_steps > 0) {
      int64_t doubles = 0;
      int64_t bs = cfg.batch_size;
      while (bs < cfg.target_batch_size) { doubles++; bs *= 2; }
      if (doubles > 0) {
        int64_t interval = cfg.batch_size_ramp_steps / doubles;
        int64_t cur_double = std::min(step / std::max(interval, int64_t(1)), doubles);
        cur_batch_size = cfg.batch_size * (1 << cur_double);
        cur_batch_size = std::min(cur_batch_size, cfg.target_batch_size);
      }
    }
    state.batch_size = cur_batch_size;

    // Epoch boundary detection
    int64_t new_epoch = dataset ? (step / steps_per_epoch) : 0;
    if (new_epoch > current_epoch && rank == 0) {
      double avg_epoch_loss = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : 0.0;
      std::cout << "--- Epoch " << current_epoch << " complete | avg_loss: "
                << std::fixed << std::setprecision(4) << avg_epoch_loss << " ---" << std::endl;
      epoch_loss_sum = 0.0;
      epoch_loss_count = 0;
      current_epoch = new_epoch;
    }

    // LR schedule
    double cur_lr = scheduler->get_lr(step, cfg.num_steps);
    scheduler->apply(*optimizer, step, cfg.num_steps);
    state.learning_rate = static_cast<float>(cur_lr);

    cb_mgr.on_step_start(state);

    optimizer->zero_grad(true);
    accum_loss_tensor.zero_();

    for (int64_t accum = 0; accum < cfg.grad_accum_steps; ++accum) {
      torch::Tensor input, labels;
      if (dataset) {
        auto [in, lab] = dataset->get_batch(cur_batch_size, device);
        input = in; labels = lab;
        dataset->prefetch_next(cur_batch_size, device);
      } else {
        input = torch::randint(0, model_cfg.vocab_size, {cur_batch_size, cur_seq_len},
                               torch::TensorOptions().dtype(torch::kLong).device(device));
        labels = torch::randint(0, model_cfg.vocab_size, {cur_batch_size, cur_seq_len},
                                torch::TensorOptions().dtype(torch::kLong).device(device));
      }

      torch::Tensor loss;
      {
        AutocastGuard ac(cfg.use_amp, device);
        loss = model->forward(input, labels, -100) / static_cast<float>(cfg.grad_accum_steps);
      }

      if (grad_scaler) {
        grad_scaler->scale(loss).backward();
      } else {
        loss.backward();
      }
      accum_loss_tensor.add_(loss.detach());
    }

    // Defer loss D2H sync: only pull from GPU when needed
    bool need_loss_sync = !callbacks.empty() ||
                          (step % cfg.log_interval == 0 && rank == 0) ||
                          (evaluator && cfg.eval_interval > 0 && (step + 1) % cfg.eval_interval == 0) ||
                          (ckpt_mgr && cfg.checkpoint_interval > 0 && (step + 1) % cfg.checkpoint_interval == 0);
    float accum_loss = 0.0f;
    if (need_loss_sync) {
      accum_loss = accum_loss_tensor.item<float>();
      epoch_loss_sum += accum_loss;
      epoch_loss_count++;
    }
    state.loss = accum_loss;
    cb_mgr.on_after_loss(state);

    // Gradient sync in distributed
    if (ddp && ddp->is_distributed()) {
      ddp->allreduce_gradients(ddp_params);
    }

    cb_mgr.on_after_backward(state);

    // Unscale + check for inf/nan if using grad scaler
    if (grad_scaler) {
      bool finite = grad_scaler->unscale_and_check(*optimizer);
      if (finite) {
        clip_grad_norm_gpu(model_params, cfg.max_grad_norm);
        grad_scaler->step(*optimizer);
      }
      grad_scaler->update();
    } else {
      clip_grad_norm_gpu(model_params, cfg.max_grad_norm);
      optimizer->step();
    }

    cb_mgr.on_after_optimizer_step(state);

    total_tokens += cur_batch_size * cur_seq_len * cfg.grad_accum_steps;

    // Update metrics
    state.metrics["loss"] = accum_loss;
    state.metrics["lr"] = static_cast<float>(cur_lr);
    state.metrics["seq_len"] = static_cast<float>(cur_seq_len);
    state.metrics["batch_size"] = static_cast<float>(cur_batch_size);
    if (grad_scaler) {
      state.metrics["grad_scale"] = grad_scaler->current_scale();
    }

    cb_mgr.on_step_end(state);

    // Logging
    if (step % cfg.log_interval == 0 && rank == 0) {
      auto now = std::chrono::steady_clock::now();
      double step_ms = std::chrono::duration<double, std::milli>(now - state.step_start).count();
      double elapsed_s = std::chrono::duration<double>(now - state.train_start).count();
      double tok_per_s = total_tokens / (elapsed_s > 0 ? elapsed_s : 1);

      std::cout << "Epoch " << current_epoch
                << " | Step " << step << "/" << cfg.num_steps
                << "  loss: " << std::fixed << std::setprecision(4) << accum_loss
                << "  lr: " << std::scientific << std::setprecision(2) << cur_lr
                << "  step_ms: " << static_cast<int>(step_ms)
                << "  tok/s: " << static_cast<int>(tok_per_s);
      if (grad_scaler) std::cout << "  scale: " << grad_scaler->current_scale();
      std::cout << std::endl;
    }

    // Evaluation
    if (evaluator && cfg.eval_interval > 0 && (step + 1) % cfg.eval_interval == 0) {
      model->eval();
      auto eval_metrics = evaluator->evaluate(*model, device);
      std::unordered_map<std::string, float> eval_float;
      for (const auto& [k, v] : eval_metrics) {
        state.metrics["eval/" + k] = static_cast<float>(v);
        eval_float[k] = static_cast<float>(v);
        if (rank == 0) std::cout << "  eval/" << k << ": " << v << std::endl;
      }
      cb_mgr.on_eval_end(state, eval_float);
      model->train();
    }

    // Checkpointing
    if (ckpt_mgr && cfg.checkpoint_interval > 0 && (step + 1) % cfg.checkpoint_interval == 0) {
      std::string tag = "step_" + std::to_string(step + 1);
      CheckpointMetadata meta;
      meta.step = step + 1;
      meta.loss = accum_loss;
      ckpt_mgr->save(tag, *model, *optimizer, meta, rank,
                      ddp ? ddp->world_size() : 1);
      ckpt_mgr->prune(cfg.keep_checkpoints);
      cb_mgr.on_checkpoint_save(state, cfg.checkpoint_dir + "/" + tag);
    }
  }

  cb_mgr.on_train_end(state);

  if (rank == 0) {
    auto end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(end - state.train_start).count();
    double avg_step_ms = total_s / cfg.num_steps * 1000.0;
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "  Steps: " << cfg.num_steps << " (" << current_epoch + 1 << " epochs)" << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Wall time: " << std::fixed << std::setprecision(2) << total_s << "s" << std::endl;
    std::cout << "  Avg step: " << std::fixed << std::setprecision(1) << avg_step_ms << "ms" << std::endl;
    std::cout << "  Throughput: " << static_cast<int>(total_tokens / total_s) << " tok/s" << std::endl;
    std::cout << "========================" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// FusedTransformer overload — delegates to same loop logic
// ---------------------------------------------------------------------------

void train(
    FusedTransformer& model,
    const TransformerConfig& model_cfg,
    const TrainConfig& cfg,
    torch::Device device,
    std::vector<std::shared_ptr<Callback>> callbacks,
    std::shared_ptr<Evaluator> evaluator) {

  model->train();

  auto ddp = DDPContext::init_from_env();
  if (ddp && ddp->is_distributed()) {
    auto params = model->parameters();
    ddp->broadcast_parameters(params);
  }
  int rank = ddp ? ddp->rank() : 0;

  std::unique_ptr<torch::optim::Optimizer> optimizer;
  if (cfg.optimizer == "lion") {
    optimizer = std::make_unique<Lion>(
        model->parameters(), LionOptions(cfg.lr).weight_decay(cfg.weight_decay));
  } else if (cfg.optimizer == "muon") {
    optimizer = std::make_unique<Muon>(
        model->parameters(), MuonOptions(cfg.lr));
  } else if (cfg.optimizer == "dion") {
    optimizer = std::make_unique<DION>(
        model->parameters(), DIONOptions(cfg.lr).weight_decay(cfg.weight_decay));
  } else if (cfg.use_foreach_optimizer) {
    optimizer = std::make_unique<ForeachAdamW>(
        model->parameters(), ForeachAdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));
  } else {
    optimizer = std::make_unique<torch::optim::AdamW>(
        model->parameters(),
        torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));
  }

  auto scheduler = create_scheduler(cfg.scheduler, cfg.lr, cfg.warmup_steps);

  std::optional<TokenDataset> dataset;
  bool gpu_data_active = false;
  if (cfg.data_path && !cfg.data_path->empty()) {
    dataset.emplace(*cfg.data_path, cfg.seq_len, true);
    dataset->reset_epoch();
    if (device.is_cuda()) {
      const int64_t cap = cfg.gpu_resident_data ? cfg.max_gpu_data_tokens : -1;
      dataset->to_device(device, cap);
      gpu_data_active = dataset->is_gpu_resident();
    } else if (cfg.gpu_resident_data) {
      dataset->to_device(device, 0);
      gpu_data_active = dataset->is_gpu_resident();
    }
  }

  if (rank == 0) {
    print_optimization_banner(cfg.optimizer, cfg.use_foreach_optimizer,
                              cfg.gpu_resident_data, gpu_data_active,
                              cfg.use_amp, true, cfg.use_cuda_graph);
  }

  // Pre-build DDP parameter list once (avoids per-step allocation)
  std::vector<torch::Tensor> ddp_params;
  if (ddp && ddp->is_distributed()) {
    for (auto& p : model->parameters()) ddp_params.push_back(p);
  }

  // Epoch tracking
  int64_t tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps;
  int64_t dataset_tokens = dataset ? dataset->size() * cfg.seq_len : 0;
  int64_t steps_per_epoch = (dataset && dataset_tokens > 0)
      ? std::max(int64_t(1), dataset_tokens / tokens_per_step) : cfg.num_steps;
  int64_t current_epoch = 0;
  double epoch_loss_sum = 0.0;
  int64_t epoch_loss_count = 0;

  auto train_start = std::chrono::steady_clock::now();
  int64_t total_tokens = 0;

  // Pre-allocate loss accumulator outside loop (avoids per-step allocation)
  torch::Tensor accum_loss_tensor = torch::zeros({}, torch::TensorOptions().device(device));

  // Cache model params once (avoids per-step vector rebuild)
  auto model_params = model->parameters();

  // ---------------------------------------------------------------------------
  // CUDA Graph: capture forward+backward as a replayable graph.
  // Eliminates per-kernel launch overhead (~5μs × 100+ kernels = 0.5-1ms/step)
  // and enables the GPU to pipeline operations without waiting for CPU dispatch.
  //
  // Requirements: CUDA device, fixed shapes (no curriculum), grad_accum=1,
  // no DDP (allreduce is cross-device). The optimizer step runs OUTSIDE the
  // graph because lr/bias_correction change per step.
  // ---------------------------------------------------------------------------
  bool graph_active = false;
#ifdef USE_CUDA
  at::cuda::CUDAGraph train_graph;
  torch::Tensor graph_input, graph_labels, graph_loss;

  bool want_graph = cfg.use_cuda_graph && device.is_cuda()
                    && (!ddp || !ddp->is_distributed());
  if (want_graph) {
    auto int_opts = torch::TensorOptions().dtype(torch::kLong).device(device);
    graph_input  = torch::empty({cfg.batch_size, cfg.seq_len}, int_opts);
    graph_labels = torch::empty({cfg.batch_size, cfg.seq_len}, int_opts);

    // ---- Warmup: populate caching allocator + cuDNN/cuBLAS benchmarks ----
    if (rank == 0) std::cout << "CUDA Graph: warming up (3 steps)...\n";
    for (int w = 0; w < 3; ++w) {
      if (dataset) {
        auto [in, lab] = dataset->get_batch(cfg.batch_size, device);
        graph_input.copy_(in);
        graph_labels.copy_(lab);
      } else {
        graph_input.copy_(torch::randint(0, model_cfg.vocab_size,
            {cfg.batch_size, cfg.seq_len}, int_opts));
        graph_labels.copy_(torch::randint(0, model_cfg.vocab_size,
            {cfg.batch_size, cfg.seq_len}, int_opts));
      }
      optimizer->zero_grad();
      {
        AutocastGuard ac(cfg.use_amp, device);
        graph_loss = model->forward(graph_input, graph_labels, -100)
                     / static_cast<float>(cfg.grad_accum_steps);
      }
      graph_loss.backward();
      clip_grad_norm_gpu(model_params, cfg.max_grad_norm);
      optimizer->step();
    }

    // ---- Capture: single micro-step (forward + backward) ----
    // The graph captures one forward+backward pass on a single micro-batch.
    // For grad_accum > 1, we replay the graph N times per optimizer step —
    // gradients accumulate naturally across replays since we only zero_grad
    // once at the start.
    if (rank == 0) {
      std::cout << "CUDA Graph: capturing forward+backward"
                << " (will replay " << cfg.grad_accum_steps << "x per step)...\n";
    }

    // Must use memset zero_grad (not set_to_none) — graph references
    // specific gradient tensor addresses that must remain valid.
    optimizer->zero_grad();

    train_graph.capture_begin();
    {
      AutocastGuard ac(cfg.use_amp, device);
      graph_loss = model->forward(graph_input, graph_labels, -100)
                   / static_cast<float>(cfg.grad_accum_steps);
    }
    graph_loss.backward();
    train_graph.capture_end();

    graph_active = true;
    if (rank == 0) std::cout << "CUDA Graph: captured successfully\n";
  }
#endif  // USE_CUDA

  for (int64_t step = 0; step < cfg.num_steps; ++step) {
    auto step_start = std::chrono::steady_clock::now();
    double cur_lr = scheduler->get_lr(step, cfg.num_steps);
    scheduler->apply(*optimizer, step, cfg.num_steps);

    // Epoch boundary detection
    int64_t new_epoch = dataset ? (step / steps_per_epoch) : 0;
    if (new_epoch > current_epoch && rank == 0) {
      double avg_epoch_loss = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : 0.0;
      std::cout << "--- Epoch " << current_epoch << " complete | avg_loss: "
                << std::fixed << std::setprecision(4) << avg_epoch_loss << " ---" << std::endl;
      epoch_loss_sum = 0.0;
      epoch_loss_count = 0;
      current_epoch = new_epoch;
    }

#ifdef USE_CUDA
    if (graph_active) {
      // ---- CUDA Graph path ----
      // Zero gradients once (memset — graph references these tensor addresses)
      optimizer->zero_grad();
      accum_loss_tensor.zero_();

      // Replay captured micro-step for each grad_accum iteration.
      // Gradients accumulate across replays (we only zero_grad once above).
      for (int64_t accum = 0; accum < cfg.grad_accum_steps; ++accum) {
        // 1. Copy fresh data into static graph buffers
        if (dataset) {
          auto [in, lab] = dataset->get_batch(cfg.batch_size, device);
          graph_input.copy_(in);
          graph_labels.copy_(lab);
        }

        // 2. Replay captured forward + backward (single graph launch)
        train_graph.replay();

        // 3. Accumulate loss
        accum_loss_tensor.add_(graph_loss.detach());
      }
    } else
#endif
    {
      // ---- Standard path (non-graph or CPU) ----
      optimizer->zero_grad(true);
      accum_loss_tensor.zero_();

      for (int64_t accum = 0; accum < cfg.grad_accum_steps; ++accum) {
        torch::Tensor input, labels;
        if (dataset) {
          auto [in, lab] = dataset->get_batch(cfg.batch_size, device);
          input = in; labels = lab;
          dataset->prefetch_next(cfg.batch_size, device);
        } else {
          input = torch::randint(0, model_cfg.vocab_size, {cfg.batch_size, cfg.seq_len},
                                 torch::TensorOptions().dtype(torch::kLong).device(device));
          labels = torch::randint(0, model_cfg.vocab_size, {cfg.batch_size, cfg.seq_len},
                                  torch::TensorOptions().dtype(torch::kLong).device(device));
        }

        torch::Tensor loss;
        {
          AutocastGuard ac(cfg.use_amp, device);
          loss = model->forward(input, labels, -100) / static_cast<float>(cfg.grad_accum_steps);
        }
        loss.backward();
        accum_loss_tensor.add_(loss.detach());
      }
    }

    // Gradient sync + optimizer step always OUTSIDE graph
    // (LR and bias correction change per step)
    if (ddp && ddp->is_distributed()) {
      ddp->allreduce_gradients(ddp_params);
    }

    clip_grad_norm_gpu(model_params, cfg.max_grad_norm);
    optimizer->step();

    total_tokens += tokens_per_step;

    // Defer D2H sync: only pull loss from GPU on log steps
    if (step % cfg.log_interval == 0 && rank == 0) {
      float accum_loss = accum_loss_tensor.item<float>();
      epoch_loss_sum += accum_loss;
      epoch_loss_count++;
      auto step_end = std::chrono::steady_clock::now();
      double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
      double elapsed_s = std::chrono::duration<double>(step_end - train_start).count();
      double tok_per_s = total_tokens / (elapsed_s > 0 ? elapsed_s : 1);

      std::cout << "Epoch " << current_epoch
                << " | Step " << step << "/" << cfg.num_steps
                << "  loss: " << std::fixed << std::setprecision(4) << accum_loss
                << "  lr: " << std::scientific << std::setprecision(2) << cur_lr
                << "  step_ms: " << static_cast<int>(step_ms)
                << "  tok/s: " << static_cast<int>(tok_per_s)
                << std::endl;
    }
  }

  if (rank == 0) {
    auto end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(end - train_start).count();
    double avg_step_ms = total_s / cfg.num_steps * 1000.0;
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "  Steps: " << cfg.num_steps << " (" << current_epoch + 1 << " epochs)" << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Wall time: " << std::fixed << std::setprecision(2) << total_s << "s" << std::endl;
    std::cout << "  Avg step: " << std::fixed << std::setprecision(1) << avg_step_ms << "ms" << std::endl;
    std::cout << "  Throughput: " << static_cast<int>(total_tokens / total_s) << " tok/s" << std::endl;
    std::cout << "========================" << std::endl;
  }
}

}  // namespace olmo_cpp
