#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <limits>

namespace n3mapping {

// Descriptor work is serial. Thread CPU excludes work by unrelated ROS and
// registration workers; wall time includes preemption and lock waits.
struct DescriptorPhaseCost {
  double wall_ms = 0.0;
  double thread_cpu_ms = 0.0;
  std::uint64_t calls = 0;
};

inline double cpuTimeMs(clockid_t clock) {
  timespec t{};
  if (clock_gettime(clock, &t) != 0)
    return std::numeric_limits<double>::quiet_NaN();
  return t.tv_sec * 1000.0 + t.tv_nsec * 1e-6;
}

class DescriptorPhaseTimer {
public:
  explicit DescriptorPhaseTimer(DescriptorPhaseCost *cost) : cost_(cost) {
    if (!cost_) return;
    wall_ = std::chrono::steady_clock::now();
    cpu_ = cpuTimeMs(CLOCK_THREAD_CPUTIME_ID);
  }
  ~DescriptorPhaseTimer() { finish(); }
  void finish() {
    if (!cost_) return;
    cost_->thread_cpu_ms += cpuTimeMs(CLOCK_THREAD_CPUTIME_ID) - cpu_;
    cost_->wall_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_).count();
    ++cost_->calls;
    cost_ = nullptr;
  }
  DescriptorPhaseTimer(const DescriptorPhaseTimer&) = delete;
  DescriptorPhaseTimer& operator=(const DescriptorPhaseTimer&) = delete;
private:
  DescriptorPhaseCost *cost_;
  std::chrono::steady_clock::time_point wall_;
  double cpu_ = 0.0;
};

struct DescriptorSearchPerformance {
  DescriptorPhaseCost total, generation, index_rebuild, coarse, place, heading;
  std::uint64_t input_points = 0, place_values = 0, heading_values = 0;
  std::uint64_t indexed_keyframes = 0, coarse_comparisons = 0;
  std::uint64_t place_comparisons = 0, heading_comparisons = 0;
  std::uint64_t returned_candidates = 0;
  std::uint64_t retrieval_places = 0;
  std::uint64_t cached_frames = 0;
  std::uint64_t retrieval_views = 0;
};

} // namespace n3mapping
