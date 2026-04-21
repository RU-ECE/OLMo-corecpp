// zwt_pretrain — production pretraining entry point for the Zero-Wait Trainer.
//
// Responsibilities:
//   * parse a TrainConfig from an INI file
//   * build the full Transformer on the selected device
//   * wire up TokenLoader (zero-stall) and fused AdamW
//   * run the step loop with gradient accumulation, cosine LR schedule, and
//     global gradient clipping
//   * log loss / tok/s / LR at configurable intervals
//   * save/resume from binary checkpoints
//
// Usage:
//   zwt_pretrain <config.ini> [--resume <ckpt.bin>]
//
// This is the binary the 3B OpenWebText run should invoke. It is deliberately
// small because everything non-trivial is in the library — this is just the
// orchestration.

#include "zwt/core/allocator.hpp"
#include "zwt/core/determinism.hpp"
#include "zwt/core/stream.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/data/token_loader.hpp"
#include "zwt/layers/module.hpp"
#include "zwt/layers/transformer.hpp"
#include "zwt/ops/elementwise.hpp"
#include "zwt/ops/xent.hpp"
#include "zwt/optim/adamw.hpp"
#include "zwt/optim/grad_clip.hpp"
#include "zwt/optim/lr_schedule.hpp"
#include "zwt/train/checkpoint.hpp"
#include "zwt/train/config.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace zwt;

namespace {

struct CliArgs {
  std::string config_path;
  std::string resume_override;
  std::string metrics_csv;
  bool        dry_run = false;
};

CliArgs parse_cli(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "usage: %s <config.ini> [--resume <ckpt.bin>] "
        "[--metrics-csv <path>] [--dry-run]\n", argv[0]);
    std::exit(2);
  }
  CliArgs a;
  a.config_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--resume" && i + 1 < argc) {
      a.resume_override = argv[++i];
    } else if (arg == "--metrics-csv" && i + 1 < argc) {
      a.metrics_csv = argv[++i];
    } else if (arg == "--dry-run") {
      a.dry_run = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      std::exit(2);
    }
  }
  return a;
}

// Pull a single float from a device scalar buffer. One host sync per call.
// Used only at logging intervals — never in the hot loop.
float pull_scalar(const Tensor& t) {
  float h = 0.f;
  if (t.device().is_cuda()) {
#ifdef USE_CUDA
    cudaMemcpy(&h, t.data(), sizeof(float), cudaMemcpyDeviceToHost);
#endif
  } else {
    h = *t.as<float>();
  }
  return h;
}

// Reset LR on every param of the optimizer (AdamW takes a single LR from
// cfg_.lr; we rewrite it in-place each step).
void apply_lr(optim::AdamW& opt, float lr) {
  opt.config().lr = lr;
}

int64_t count_params(const std::vector<Parameter*>& ps) {
  int64_t n = 0;
  for (auto* p : ps) n += p->numel();
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  CliArgs cli = parse_cli(argc, argv);

  train::TrainConfig cfg = train::load_train_config(cli.config_path);
  if (!cli.resume_override.empty()) {
    cfg.resume_from = cli.resume_override;
  }

  // Must run before the first cuBLAS handle is created — the workspace
  // config is consulted only at handle creation.
  if (cfg.deterministic) {
    set_deterministic(true);
    init_determinism_env();
    std::fprintf(stderr, "determinism: ON (cuBLAS workspace :4096:8, "
                         "atomic-free reductions)\n");
  }

#ifdef USE_CUDA
  Device dev = Device::cuda(0);
  DType  param_dtype = DType::BF16;
#else
  Device dev = Device::cpu();
  DType  param_dtype = DType::F32;
#endif

  std::fprintf(stderr, "=== zwt_pretrain ===\n");
  std::fprintf(stderr, "config: %s\n", cli.config_path.c_str());
  std::fprintf(stderr, "device: %s  dtype: %s\n",
               dev.is_cuda() ? "cuda:0" : "cpu", dtype_name(param_dtype));
  std::fprintf(stderr, "model:  vocab=%lld d_model=%lld n_layers=%lld heads=%lld d_ffn=%lld seq=%lld\n",
               (long long)cfg.model.vocab_size,
               (long long)cfg.model.d_model,
               (long long)cfg.model.n_layers,
               (long long)cfg.model.n_heads,
               (long long)cfg.model.d_ffn,
               (long long)cfg.model.max_seq);
  std::fprintf(stderr, "data:   %s  seq=%lld batch=%lld grad_accum=%lld\n",
               cfg.data_path.c_str(),
               (long long)cfg.seq_len,
               (long long)cfg.batch_size,
               (long long)cfg.grad_accum);

  // Size the activation arena generously — the forward pass of a block needs
  // a few hundred MB of scratch at seq=2048. The caller controls the cap via
  // [runtime] arena_mb.
  size_t arena_bytes = static_cast<size_t>(std::max<int64_t>(cfg.arena_mb, 256)) << 20;
  set_activation_arena_capacity(arena_bytes);
  std::fprintf(stderr, "arena:  %zu MiB\n", arena_bytes >> 20);

  // Build the model and optimizer. The Transformer seeds each block / proj
  // with a distinct salt so no two parameters start with identical values.
  Transformer model(cfg.model, param_dtype, dev, cfg.init_seed);
  std::vector<Parameter*> params;
  model.collect_params(params);
  std::fprintf(stderr, "params: %lld (%.2f M)\n",
               (long long)count_params(params),
               double(count_params(params)) / 1e6);

  optim::AdamW opt(params, cfg.adamw);

  // Optional resume. When restoring, the checkpoint meta tells us which step
  // to pick up at and which data cursor to seed the loader with.
  int64_t  resume_step = 0;
  int64_t  resume_cursor = 0;
  if (!cfg.resume_from.empty()) {
    std::fprintf(stderr, "resuming from %s\n", cfg.resume_from.c_str());
    train::CheckpointMeta m = train::load_checkpoint(cfg.resume_from, params, opt);
    resume_step   = m.step;
    resume_cursor = m.data_cursor;
    std::fprintf(stderr, "  resumed at step %lld, cursor %lld\n",
                 (long long)resume_step, (long long)resume_cursor);
  }

  // Build the token loader and start its producer thread.
  data::TokenLoader::Options dopts;
  dopts.path         = cfg.data_path;
  dopts.seq_len      = cfg.seq_len;
  dopts.batch_size   = cfg.batch_size;
  dopts.shuffle      = cfg.shuffle;
  dopts.seed         = cfg.data_seed;
  dopts.device       = dev;
  dopts.start_cursor = resume_cursor;
  data::TokenLoader loader(dopts);
  loader.start();
  std::fprintf(stderr, "loader: %lld chunks, %lld steps/epoch\n",
               (long long)((loader.steps_per_epoch() * cfg.batch_size)),
               (long long)loader.steps_per_epoch());

  if (cli.dry_run) {
    std::fprintf(stderr, "dry-run: exiting before step loop\n");
    return 0;
  }

  // Optional machine-readable metrics CSV. Consumed by
  // zwt/scripts/loss_curve_check.py to verify we track a published
  // baseline. Columns are stable: step,loss,lr,grad_norm,tokens_seen,
  // wall_secs,tok_per_s. Append-mode when resuming so the curve is
  // continuous across restarts.
  std::FILE* metrics_fp = nullptr;
  if (!cli.metrics_csv.empty()) {
    const bool fresh = cli.resume_override.empty();
    metrics_fp = std::fopen(cli.metrics_csv.c_str(), fresh ? "w" : "a");
    if (!metrics_fp) {
      std::fprintf(stderr, "warning: could not open %s for metrics\n",
                   cli.metrics_csv.c_str());
    } else if (fresh) {
      std::fprintf(metrics_fp,
          "step,loss,lr,grad_norm,tokens_seen,wall_secs,tok_per_s\n");
      std::fflush(metrics_fp);
    }
  }

  // Hot loop.
  const int64_t batch = cfg.batch_size;
  const int64_t seq   = cfg.seq_len;
  const int64_t vocab = cfg.model.vocab_size;
  const int64_t max_steps = cfg.max_steps;

  auto t_start = std::chrono::steady_clock::now();
  int64_t tokens_seen = 0;
  float last_loss = 0.f;

  for (int64_t step = resume_step + 1; step <= max_steps; ++step) {
    // Set LR from the cosine schedule (step is 1-indexed).
    float lr = cfg.schedule.lr_at(step);
    apply_lr(opt, lr);

    opt.zero_grad();

    // Gradient accumulation: sum grads from `grad_accum` micro-batches before
    // a single optimizer step. The loss we log is the mean over those
    // micro-batches.
    float accum_loss = 0.f;
    for (int64_t acc = 0; acc < cfg.grad_accum; ++acc) {
      step_begin();               // reset activation arena

      auto batch_data = loader.next();
      Tensor logits = model.forward(batch_data.input);               // [B,S,V]
      Shape logits2d{batch * seq, vocab};
      Tensor logits_flat = logits.view(logits2d);
      Tensor tgt_flat    = batch_data.target.view({batch * seq});

      Tensor loss = empty_scratch({1}, DType::F32, dev);
      Tensor grad_logits = empty_scratch(logits2d, param_dtype, dev);
      ops::cross_entropy(logits_flat, tgt_flat, loss, &grad_logits, /*ignore=*/-100);

      // For grad accumulation we want the effective batch to be
      // batch_size * grad_accum while taking a single optimizer step. Scale
      // the loss grad by 1/grad_accum so each micro-batch contributes its
      // share. Do NOT rescale LR — AdamW is not linear in the gradient
      // (the sqrt(v)+eps denominator breaks linearity).
      if (cfg.grad_accum > 1) {
        ops::scale(grad_logits, 1.0f / float(cfg.grad_accum));
      }

      model.backward(grad_logits.view(logits.shape()));

      if (step % cfg.log_interval == 0 || step == 1 || acc == cfg.grad_accum - 1) {
        accum_loss += pull_scalar(loss);
      }
      tokens_seen += batch * seq;
    }
    // Average loss across accumulated micro-batches (for logging only).
    last_loss = accum_loss / float(std::max<int64_t>(cfg.grad_accum, 1));

    // Clip and step.
    float gnorm = 0.f;
    if (cfg.grad_clip > 0.f) {
      gnorm = optim::clip_grad_norm(params, cfg.grad_clip);
    }
    opt.step();

    // Log and checkpoint.
    if (step % cfg.log_interval == 0 || step == 1) {
      auto t_now = std::chrono::steady_clock::now();
      double secs = std::chrono::duration<double>(t_now - t_start).count();
      double tps = tokens_seen / std::max(secs, 1e-9);
      std::fprintf(stderr,
          "step %6lld  loss %.4f  lr %.2e  |g| %.3f  %.0f tok/s\n",
          (long long)step, last_loss, lr, gnorm, tps);
      if (metrics_fp) {
        std::fprintf(metrics_fp,
            "%lld,%.6f,%.6e,%.6f,%lld,%.3f,%.3f\n",
            (long long)step, last_loss, lr, gnorm,
            (long long)tokens_seen, secs, tps);
        std::fflush(metrics_fp);
      }
    }

    if (cfg.ckpt_interval > 0 && step % cfg.ckpt_interval == 0) {
      train::CheckpointMeta meta;
      meta.step        = step;
      meta.seed        = cfg.init_seed;
      meta.data_cursor = loader.cursor();
      meta.lr          = lr;
      meta.loss        = last_loss;
      std::fprintf(stderr, "  writing ckpt %s (step %lld)\n",
                   cfg.ckpt_path.c_str(), (long long)step);
      train::save_checkpoint(cfg.ckpt_path, params, opt, meta);
    }
  }

#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif

  // Final checkpoint on clean exit.
  if (!cfg.ckpt_path.empty()) {
    train::CheckpointMeta meta;
    meta.step        = max_steps;
    meta.seed        = cfg.init_seed;
    meta.data_cursor = loader.cursor();
    meta.lr          = cfg.schedule.lr_at(max_steps);
    meta.loss        = last_loss;
    std::fprintf(stderr, "writing final ckpt %s\n", cfg.ckpt_path.c_str());
    train::save_checkpoint(cfg.ckpt_path, params, opt, meta);
  }
  if (metrics_fp) std::fclose(metrics_fp);
  std::fprintf(stderr, "done.\n");
  return 0;
}
