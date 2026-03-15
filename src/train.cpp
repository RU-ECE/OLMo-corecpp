#include "olmo_cpp/train.hpp"
#include "olmo_cpp/data/token_dataset.hpp"
#include "olmo_cpp/distributed/ddp.hpp"
#include "olmo_cpp/optim/scheduler.hpp"
#include "olmo_cpp/train/callback.hpp"
#include "olmo_cpp/train/grad_scaler.hpp"
#include "olmo_cpp/train/activation_checkpoint.hpp"
#include "olmo_cpp/train/checkpoint.hpp"
#include "olmo_cpp/eval/evaluator.hpp"
#include "olmo_cpp/io/filesystem.hpp"
#include <torch/nn/init.h>
#include <iostream>
#include <cmath>
#include <chrono>

namespace olmo_cpp {

namespace {
double cosine_warmup_lr(int64_t step, int64_t warmup_steps, double base_lr, int64_t total_steps) {
  if (step < warmup_steps) {
    return base_lr * static_cast<double>(step + 1) / static_cast<double>(warmup_steps);
  }
  double progress = static_cast<double>(step - warmup_steps) / static_cast<double>(total_steps - warmup_steps);
  return 0.5 * base_lr * (1.0 + std::cos(M_PI * progress));
}
}  // namespace

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
    bool use_amp) {
  (void)use_amp;
  model->train();

  auto optimizer = torch::optim::AdamW(
      model->parameters(), torch::optim::AdamWOptions(lr).weight_decay(0.01));
  std::cout << "Using AdamW optimizer (lr=" << lr << ")" << std::endl;

  auto ddp = DDPContext::init_from_env();
  if (ddp && ddp->is_distributed()) {
    auto params = model->parameters();
    ddp->broadcast_parameters(params);
  }

  std::optional<TokenDataset> dataset;
  if (data_path && !data_path->empty()) {
    dataset.emplace(*data_path, seq_len, true);
    dataset->reset_epoch();
  }

  auto train_start = std::chrono::steady_clock::now();
  int64_t total_tokens = 0;

  for (int64_t step = 0; step < num_steps; ++step) {
    auto step_start = std::chrono::steady_clock::now();
    double step_lr = cosine_warmup_lr(step, warmup_steps, lr, num_steps);
    static_cast<torch::optim::AdamWOptions&>(optimizer.param_groups()[0].options()).lr(step_lr);

    optimizer.zero_grad();
    float accum_loss = 0.0f;

    for (int64_t accum = 0; accum < grad_accum_steps; ++accum) {
      torch::Tensor input, labels;
      if (dataset) {
        auto [in, lab] = dataset->get_batch(batch_size, device);
        input = in; labels = lab;
      } else {
        input = torch::randint(0, cfg.vocab_size, {batch_size, seq_len},
                               torch::TensorOptions().dtype(torch::kLong).device(device));
        labels = torch::randint(0, cfg.vocab_size, {batch_size, seq_len},
                                torch::TensorOptions().dtype(torch::kLong).device(device));
      }

      auto loss = model->forward(input, labels, -100) / static_cast<float>(grad_accum_steps);
      loss.backward();
      accum_loss += loss.item<float>();
    }

    if (ddp && ddp->is_distributed()) {
      std::vector<torch::Tensor> params;
      for (auto& p : model->parameters()) params.push_back(p);
      ddp->allreduce_gradients(params);
    }

    torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
    optimizer.step();

    total_tokens += batch_size * seq_len * grad_accum_steps;

    if (step % 10 == 0 && (!ddp || ddp->rank() == 0)) {
      auto step_end = std::chrono::steady_clock::now();
      double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
      double elapsed_s = std::chrono::duration<double>(step_end - train_start).count();
      double tok_per_s = total_tokens / (elapsed_s > 0 ? elapsed_s : 1);
      double eta_s = (num_steps - step) * (elapsed_s / (step > 0 ? step : 1));

      std::cout << "Step " << step << "/" << num_steps
                << "  loss: " << accum_loss
                << "  lr: " << step_lr
                << "  step_ms: " << static_cast<int>(step_ms)
                << "  tok/s: " << static_cast<int>(tok_per_s)
                << "  ETA: " << static_cast<int>(eta_s / 60) << "m"
                << std::endl;
    }
  }

  if (!ddp || ddp->rank() == 0) {
    auto end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(end - train_start).count();
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "  Steps: " << num_steps << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Wall time: " << static_cast<int>(total_s) << "s ("
              << static_cast<int>(total_s / 60) << "m " << static_cast<int>(total_s) % 60 << "s)" << std::endl;
    std::cout << "  Throughput: " << static_cast<int>(total_tokens / total_s) << " tok/s" << std::endl;
    std::cout << "========================" << std::endl;
  }
}

// Full-featured train() with all infrastructure
void train(
    Transformer& model,
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

  auto optimizer = std::make_unique<torch::optim::AdamW>(
      model->parameters(),
      torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));

  auto scheduler = create_scheduler(cfg.scheduler, cfg.lr, cfg.warmup_steps);

  std::optional<GradScaler> grad_scaler;
  if (cfg.use_grad_scaler) {
    grad_scaler.emplace();
  }

  std::optional<CheckpointManager> ckpt_mgr;
  if (!cfg.checkpoint_dir.empty()) {
    ckpt_mgr.emplace(cfg.checkpoint_dir);
  }

  std::optional<TokenDataset> dataset;
  if (cfg.data_path && !cfg.data_path->empty()) {
    dataset.emplace(*cfg.data_path, cfg.seq_len, true);
    dataset->reset_epoch();
  }

  CallbackManager cb_mgr;
  for (auto& cb : callbacks) cb_mgr.add(cb);

  TrainState state;
  state.train_start = std::chrono::steady_clock::now();
  cb_mgr.on_train_start(state);

  for (int64_t step = 0; step < cfg.num_steps; ++step) {
    state.step_start = std::chrono::steady_clock::now();
    state.global_step = step;

    // Sequence length scheduling
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

    double cur_lr = scheduler->get_lr(step, cfg.num_steps);
    scheduler->apply(*optimizer, step, cfg.num_steps);
    state.learning_rate = static_cast<float>(cur_lr);

    cb_mgr.on_step_start(state);

    optimizer->zero_grad();
    float accum_loss = 0.0f;

    for (int64_t accum = 0; accum < cfg.grad_accum_steps; ++accum) {
      torch::Tensor input, labels;
      if (dataset) {
        auto [in, lab] = dataset->get_batch(cur_batch_size, device);
        input = in; labels = lab;
      } else {
        input = torch::randint(0, model_cfg.vocab_size, {cur_batch_size, cur_seq_len},
                               torch::TensorOptions().dtype(torch::kLong).device(device));
        labels = torch::randint(0, model_cfg.vocab_size, {cur_batch_size, cur_seq_len},
                                torch::TensorOptions().dtype(torch::kLong).device(device));
      }

      auto loss = model->forward(input, labels, -100) / static_cast<float>(cfg.grad_accum_steps);

      if (grad_scaler) {
        grad_scaler->scale(loss).backward();
      } else {
        loss.backward();
      }
      accum_loss += loss.item<float>();
    }

    state.loss = accum_loss;
    cb_mgr.on_after_loss(state);

    if (ddp && ddp->is_distributed()) {
      std::vector<torch::Tensor> params;
      for (auto& p : model->parameters()) params.push_back(p);
      ddp->allreduce_gradients(params);
    }

    cb_mgr.on_after_backward(state);

    if (grad_scaler) {
      bool finite = grad_scaler->unscale_and_check(*optimizer);
      if (finite) {
        torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.max_grad_norm);
        grad_scaler->step(*optimizer);
      }
      grad_scaler->update();
    } else {
      torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.max_grad_norm);
      optimizer->step();
    }

    cb_mgr.on_after_optimizer_step(state);

    state.metrics["loss"] = accum_loss;
    state.metrics["lr"] = static_cast<float>(cur_lr);
    state.metrics["seq_len"] = static_cast<float>(cur_seq_len);
    state.metrics["batch_size"] = static_cast<float>(cur_batch_size);
    if (grad_scaler) {
      state.metrics["grad_scale"] = grad_scaler->current_scale();
    }

    cb_mgr.on_step_end(state);

    if (step % 10 == 0 && rank == 0) {
      std::cout << "Step " << step
                << " loss: " << accum_loss
                << " lr: " << cur_lr
                << " seq_len: " << cur_seq_len;
      if (grad_scaler) std::cout << " scale: " << grad_scaler->current_scale();
      std::cout << std::endl;
    }

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
    std::cout << "Training complete." << std::endl;
  }
}

}  // namespace olmo_cpp
