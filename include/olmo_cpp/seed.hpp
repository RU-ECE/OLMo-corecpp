#pragma once

#include <cstdint>
#include <string>
#include <random>
#include <optional>
#include <torch/torch.h>

namespace olmo_cpp {

/// Global seed state — mirrors OLMo-core's seed_all() from utils.py.
///
/// Seeds:
///   - torch CPU RNG         (torch::manual_seed)
///   - torch CUDA RNG        (torch::cuda::manual_seed_all, if CUDA available)
///   - std::mt19937           (C++ stdlib, used for data shuffling)
///   - std::random_device     (not seedable, but we record it)
///
/// Prints the active seed so training runs are always reproducible.
/// If no seed is provided, generates one from std::random_device and prints it.

struct SeedState {
  uint64_t seed;
  bool was_explicit;  // true if user provided --seed, false if auto-generated

  /// The global C++ PRNG, seeded deterministically
  std::mt19937_64 rng;

  /// A torch::Generator seeded with the same seed, for weight init etc.
  torch::Generator torch_gen;
};

/// Seed all random number generators. Call this once at startup.
/// If seed == std::nullopt, a random seed is generated and printed.
/// Returns the SeedState for further use (e.g., passing to init_weights).
SeedState seed_all(std::optional<uint64_t> seed = std::nullopt);

/// Get the global seed state (set by seed_all). Throws if seed_all not called.
SeedState& global_seed_state();

/// Print a summary of all RNG states (for debugging reproducibility).
void print_rng_state_summary();

/// Save RNG states to a checkpoint-compatible format.
/// Captures: torch CPU state, torch CUDA state (if available), mt19937 state.
struct RNGCheckpoint {
  uint64_t original_seed;
  torch::Tensor torch_cpu_state;
  std::optional<torch::Tensor> torch_cuda_state;
  std::string mt19937_state;  // serialized as string
};

RNGCheckpoint capture_rng_state();
void restore_rng_state(const RNGCheckpoint& checkpoint);

}  // namespace olmo_cpp
