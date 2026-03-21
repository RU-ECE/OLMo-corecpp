#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <mutex>
#include <cmath>
#include <torch/torch.h>

namespace olmo_cpp {

/// Lightweight profiler for understanding where time is spent in training/inference.
/// Thread-safe. Designed to be always-on with minimal overhead (<0.1%).
///
/// Usage:
///   {
///     ProfileScope scope("attention");
///     // ... attention code ...
///   }
///
///   // Or manually:
///   profiler().start("ffn");
///   // ... ffn code ...
///   profiler().stop("ffn");
///
///   // At end of training:
///   profiler().report();

struct TimingStats {
  int64_t count = 0;
  double total_us = 0.0;     // microseconds
  double min_us = 1e18;
  double max_us = 0.0;
  double sum_sq_us = 0.0;    // for variance

  double mean_us() const { return count > 0 ? total_us / count : 0.0; }
  double std_us() const {
    if (count < 2) return 0.0;
    double mean = mean_us();
    return std::sqrt(sum_sq_us / count - mean * mean);
  }
  double total_ms() const { return total_us / 1000.0; }
  double mean_ms() const { return mean_us() / 1000.0; }
};

class Profiler {
 public:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;

  void start(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_[name] = Clock::now();
  }

  void stop(const std::string& name) {
    auto end = Clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_.find(name);
    if (it == active_.end()) return;

    double us = std::chrono::duration<double, std::micro>(end - it->second).count();
    active_.erase(it);

    auto& stats = stats_[name];
    stats.count++;
    stats.total_us += us;
    stats.min_us = std::min(stats.min_us, us);
    stats.max_us = std::max(stats.max_us, us);
    stats.sum_sq_us += us * us;
  }

  void record(const std::string& name, double us) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& stats = stats_[name];
    stats.count++;
    stats.total_us += us;
    stats.min_us = std::min(stats.min_us, us);
    stats.max_us = std::max(stats.max_us, us);
    stats.sum_sq_us += us * us;
  }

  /// Get stats for a specific region
  TimingStats get(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_.find(name);
    if (it != stats_.end()) return it->second;
    return {};
  }

  /// Print a formatted report sorted by total time
  void report(const std::string& title = "Profile Report") const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stats_.empty()) {
      std::cout << "[Profiler] No data collected.\n";
      return;
    }

    // Sort by total time descending
    std::vector<std::pair<std::string, TimingStats>> sorted(stats_.begin(), stats_.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.total_us > b.second.total_us; });

    double total_wall_us = 0.0;
    for (const auto& [name, s] : sorted) total_wall_us += s.total_us;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────────┐\n"
              << "│ " << std::left << std::setw(78) << title << "│\n"
              << "├──────────────────────┬────────┬──────────┬──────────┬──────────┬───────────────┤\n"
              << "│ Region               │ Calls  │ Total ms │ Mean ms  │ Std ms   │ % of Total    │\n"
              << "├──────────────────────┼────────┼──────────┼──────────┼──────────┼───────────────┤\n";

    for (const auto& [name, s] : sorted) {
      double pct = total_wall_us > 0 ? (s.total_us / total_wall_us * 100.0) : 0.0;
      std::string bar(static_cast<size_t>(pct / 5.0), '#');

      std::cout << "│ " << std::left << std::setw(21) << name.substr(0, 21)
                << "│ " << std::right << std::setw(6) << s.count
                << " │ " << std::right << std::setw(8) << std::fixed << std::setprecision(1) << s.total_ms()
                << " │ " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << s.mean_ms()
                << " │ " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << (s.std_us() / 1000.0)
                << " │ " << std::right << std::setw(5) << std::fixed << std::setprecision(1) << pct
                << "% " << std::left << std::setw(7) << bar
                << "│\n";
    }

    std::cout << "├──────────────────────┴────────┴──────────┴──────────┴──────────┴───────────────┤\n"
              << "│ Total wall time: " << std::fixed << std::setprecision(1)
              << std::setw(10) << (total_wall_us / 1000.0) << " ms"
              << std::setw(49) << " " << "│\n"
              << "└─────────────────────────────────────────────────────────────────────────────────┘\n"
              << std::endl;
  }

  /// Reset all collected data
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.clear();
    active_.clear();
  }

  /// Get all stats (for programmatic access)
  std::unordered_map<std::string, TimingStats> all_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, TimingStats> stats_;
  std::unordered_map<std::string, TimePoint> active_;
};

/// Global profiler instance
Profiler& profiler();

/// RAII scope timer — measures the lifetime of the object
class ProfileScope {
 public:
  explicit ProfileScope(const std::string& name, Profiler& p = profiler())
      : name_(name), profiler_(p) {
    profiler_.start(name_);
  }
  ~ProfileScope() {
    profiler_.stop(name_);
  }
  ProfileScope(const ProfileScope&) = delete;
  ProfileScope& operator=(const ProfileScope&) = delete;

 private:
  std::string name_;
  Profiler& profiler_;
};

/// Memory tracking utilities
struct MemoryStats {
  int64_t allocated_bytes = 0;
  int64_t reserved_bytes = 0;
  int64_t peak_allocated_bytes = 0;
};

/// Get current GPU memory stats (CUDA or MPS)
MemoryStats get_memory_stats(torch::Device device);

/// Print memory summary
void print_memory_summary(torch::Device device);

}  // namespace olmo_cpp
