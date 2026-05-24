/**
 * src/train_wave.cpp
 *
 * Training loop for WaveTransformer (wave-field / Hyena SSM).
 * Uses the same optimizer / LR-scheduler / dataset infrastructure
 * as train.cpp but routes forward() through WaveTransformer.
 * Single-GPU research path — no CUDA graphs, no DDP.
 */
#include "olmo_cpp/train.hpp"
#include "olmo_cpp/data/token_dataset.hpp"
#include "olmo_cpp/optim/lion.hpp"
#include "olmo_cpp/optim/muon.hpp"
#include "olmo_cpp/optim/dion.hpp"
#include "olmo_cpp/optim/foreach_adamw.hpp"
#include "olmo_cpp/optim/grad_clip.hpp"
#include "olmo_cpp/optim/scheduler.hpp"
#include "olmo_cpp/train/grad_scaler.hpp"
#include <ATen/autocast_mode.h>
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

namespace olmo_cpp {

void train(
    WaveTransformer& model,
    const TransformerConfig& model_cfg,
    const TrainConfig& cfg,
    torch::Device device,
    std::vector<std::shared_ptr<Callback>> /*callbacks*/,
    std::shared_ptr<Evaluator> /*evaluator*/)
{
  model->train();
  model->to(device);

  // ── Optimizer ──────────────────────────────────────────────────────────────
  auto params = model->parameters();
  std::unique_ptr<torch::optim::Optimizer> optimizer;
  if (cfg.optimizer == "lion") {
    optimizer = std::make_unique<Lion>(
        params, LionOptions(cfg.lr).weight_decay(cfg.weight_decay));
  } else if (cfg.optimizer == "muon") {
    optimizer = std::make_unique<Muon>(params, MuonOptions(cfg.lr));
  } else if (cfg.optimizer == "dion") {
    optimizer = std::make_unique<DION>(
        params, DIONOptions(cfg.lr).weight_decay(cfg.weight_decay));
  } else {
    optimizer = std::make_unique<torch::optim::AdamW>(
        params,
        torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));
  }

  // ── LR scheduler ───────────────────────────────────────────────────────────
  auto scheduler = create_scheduler(cfg.scheduler, cfg.lr, cfg.warmup_steps);

  // ── Dataset ────────────────────────────────────────────────────────────────
  std::optional<TokenDataset> dataset;
  if (cfg.data_path && !cfg.data_path->empty()) {
    dataset.emplace(*cfg.data_path, cfg.seq_len, /*shuffle=*/true);
    dataset->reset_epoch();
    if (device.is_cuda()) dataset->to_device(device, 0);
  }

  // ── Grad scaler (AMP / FP16) ───────────────────────────────────────────────
  std::optional<GradScaler> grad_scaler;
  if (cfg.use_grad_scaler) grad_scaler.emplace();

  // ── Parameter count ────────────────────────────────────────────────────────
  int64_t n_params = 0;
  for (auto& p : params) n_params += p.numel();
  std::cout << "[WaveTransformer] "
            << (n_params / 1'000'000) << "M params"
            << "  d=" << model_cfg.d_model
            << "  layers=" << model_cfg.n_layers
            << "  mixer=";
  switch (model_cfg.sequence_mixer) {
    case TransformerConfig::SequenceMixerType::Wave:        std::cout << "wave\n"; break;
    case TransformerConfig::SequenceMixerType::Hyena:       std::cout << "hyena\n"; break;
    case TransformerConfig::SequenceMixerType::HybridWave:  std::cout << "hybrid_wave\n"; break;
    case TransformerConfig::SequenceMixerType::HybridHyena: std::cout << "hybrid_hyena\n"; break;
    default: std::cout << "?\n"; break;
  }

  const int64_t log_every = cfg.log_interval > 0 ? cfg.log_interval : 50;
  double smooth_loss = 0.0;
  auto t_start = std::chrono::steady_clock::now();

  // ── Training loop ──────────────────────────────────────────────────────────
  for (int64_t step = 0; step < cfg.num_steps; ++step) {
    // Apply LR schedule
    scheduler->apply(*optimizer, step, cfg.num_steps);
    double cur_lr = scheduler->get_lr(step, cfg.num_steps);

    optimizer->zero_grad();
    torch::Tensor accum_loss = torch::zeros({}, torch::TensorOptions().device(device));

    for (int64_t acc = 0; acc < cfg.grad_accum_steps; ++acc) {
      // ── Sample batch ────────────────────────────────────────────────────────
      torch::Tensor input, labels;
      if (dataset) {
        std::tie(input, labels) = dataset->get_batch(cfg.batch_size, device);
      } else {
        // Synthetic smoke-test batch (no data file required)
        input  = torch::randint(0, model_cfg.vocab_size,
                                {cfg.batch_size, cfg.seq_len},
                                torch::TensorOptions().dtype(torch::kLong).device(device));
        labels = input.clone();
      }

      // ── Forward + loss ──────────────────────────────────────────────────────
      torch::Tensor loss;
      if (cfg.use_amp && device.is_cuda()) {
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        loss = model->forward(input, labels) / static_cast<float>(cfg.grad_accum_steps);
        at::autocast::set_autocast_enabled(at::kCUDA, false);
      } else {
        loss = model->forward(input, labels) / static_cast<float>(cfg.grad_accum_steps);
      }

      // ── Backward ────────────────────────────────────────────────────────────
      if (grad_scaler) {
        grad_scaler->scale(loss).backward();
      } else {
        loss.backward();
      }
      accum_loss += loss.detach();
    }

    // ── Gradient clip + optimizer step ──────────────────────────────────────
    if (grad_scaler) {
      bool finite = grad_scaler->unscale_and_check(*optimizer);
      if (finite) {
        clip_grad_norm_gpu(model->parameters(), cfg.max_grad_norm);
        grad_scaler->step(*optimizer);
      }
      grad_scaler->update();
    } else {
      clip_grad_norm_gpu(model->parameters(), cfg.max_grad_norm);
      optimizer->step();
    }

    // ── Logging ─────────────────────────────────────────────────────────────
    float loss_val = accum_loss.item<float>();
    smooth_loss = (step == 0) ? loss_val : 0.9 * smooth_loss + 0.1 * loss_val;

    if (step % log_every == 0) {
      auto now  = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - t_start).count();
      double tok_per_sec = static_cast<double>(step + 1) *
                           cfg.batch_size * cfg.seq_len /
                           std::max(elapsed, 1e-9);
      std::cout << std::fixed << std::setprecision(4)
                << "step=" << std::setw(6) << step
                << "  loss=" << loss_val
                << "  smooth=" << smooth_loss
                << "  ppl=" << std::exp(smooth_loss)
                << "  lr=" << std::scientific << std::setprecision(2) << cur_lr
                << "  tok/s=" << std::fixed << std::setprecision(0) << tok_per_sec
                << "\n";
    }

    // Prefetch next batch while GPU processes the current step
    if (dataset) dataset->prefetch_next(cfg.batch_size, device);
  }

  std::cout << "[WaveTransformer] Training complete. "
            << "Final smooth loss=" << smooth_loss
            << "  ppl=" << std::exp(smooth_loss) << "\n";
}

}  // namespace olmo_cpp
