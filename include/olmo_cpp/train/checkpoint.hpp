#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <optional>
#include <unordered_map>

namespace olmo_cpp {

class FileSystem;

struct CheckpointMetadata {
  int64_t step = 0;
  int64_t epoch = 0;
  double loss = 0.0;
  std::string model_config_json;
  std::string optimizer_type;
  int world_size = 1;
  int rank = 0;
};

/// Distributed checkpoint manager with multi-threaded I/O
class CheckpointManager {
 public:
  /// \param base_path Local or remote URI (s3://, gs://, file://)
  /// \param num_io_threads Number of threads for parallel tensor writes
  explicit CheckpointManager(const std::string& base_path, int num_io_threads = 4);

  /// Save model + optimizer state + metadata. Each rank writes its own shard.
  void save(const std::string& tag,
            const torch::nn::Module& model,
            const torch::optim::Optimizer& optimizer,
            const CheckpointMetadata& meta,
            int rank = 0, int world_size = 1);

  /// Non-blocking save: returns a future that completes when I/O finishes
  std::future<void> save_async(const std::string& tag,
                                const torch::nn::Module& model,
                                const torch::optim::Optimizer& optimizer,
                                const CheckpointMetadata& meta,
                                int rank = 0, int world_size = 1);

  /// Load model + optimizer from checkpoint tag
  CheckpointMetadata load(const std::string& tag,
                          torch::nn::Module& model,
                          torch::optim::Optimizer& optimizer,
                          int rank = 0, int world_size = 1);

  /// Find the latest checkpoint tag by step number
  std::optional<std::string> latest() const;

  /// Keep only the N most recent checkpoints
  void prune(int keep_n);

  /// List all checkpoint tags
  std::vector<std::string> list_checkpoints() const;

 private:
  std::unique_ptr<FileSystem> fs_;
  std::string base_path_;
  int num_io_threads_;

  std::string shard_dir(const std::string& tag, int rank) const;
  void write_metadata(const std::string& dir, const CheckpointMetadata& meta) const;
  CheckpointMetadata read_metadata(const std::string& dir) const;
};

}  // namespace olmo_cpp
