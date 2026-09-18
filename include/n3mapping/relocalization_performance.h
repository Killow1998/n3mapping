#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>

namespace n3mapping {

// Shared by normal status and optional debug logs; unexecuted stages stay NaN.
struct TrackingPerformance {
  bool available = false;
  bool strict_loaded_map = false;
  double tracking_total_ms = std::numeric_limits<double>::quiet_NaN();
  double lock_wait_ms = std::numeric_limits<double>::quiet_NaN();
  double nearest_keyframe_ms = std::numeric_limits<double>::quiet_NaN();
  double loaded_map_cache_ms = std::numeric_limits<double>::quiet_NaN();
  double submap_build_ms = std::numeric_limits<double>::quiet_NaN();
  double target_prepare_ms = std::numeric_limits<double>::quiet_NaN();
  double source_prepare_ms = std::numeric_limits<double>::quiet_NaN();
  double registration_ms = std::numeric_limits<double>::quiet_NaN();
  double retry_registration_ms = std::numeric_limits<double>::quiet_NaN();
  double visibility_ms = std::numeric_limits<double>::quiet_NaN();
  bool loaded_map_target_cache_hit = false;
  bool loaded_map_target_cache_miss = false;
  bool localization_target_cache_enabled = false;
  bool localization_target_cache_hit = false;
  bool localization_target_cache_miss = false;
  std::size_t localization_target_cache_entry_bytes = 0;
  std::size_t localization_target_cache_total_bytes = 0;
  std::size_t localization_target_cache_entries = 0;
};

// Per-observation costs, never used by localization decisions. Zero means the
// stage did not run; available distinguishes a search attempt from tracking.
struct RelocalizationPerformance {
  bool available = false;
  double total_ms = 0.0;
  double source_prepare_ms = 0.0;
  double visibility_prepare_ms = 0.0;
  double target_build_ms = 0.0;
  double target_prepare_ms = 0.0;
  double align_ms = 0.0;
  double visibility_ms = 0.0;
  std::uint64_t source_preparations = 0;
  std::uint64_t visibility_preparations = 0;
  std::uint64_t target_builds = 0;
  std::uint64_t target_preparations = 0;
  std::uint64_t alignments = 0;
  std::uint64_t visibility_evaluations = 0;
  std::uint64_t candidates = 0;
  std::uint64_t hypotheses = 0;
  std::uint64_t cache_hits = 0;
  std::uint64_t cache_misses = 0;
  std::uint64_t cache_bytes = 0;
  std::uint64_t cache_entries = 0;
  std::uint64_t effective_cache_max_bytes = 0;
  std::uint64_t effective_cache_max_entries = 0;

  void addCandidate(const RelocalizationPerformance &other) {
    target_build_ms += other.target_build_ms;
    target_prepare_ms += other.target_prepare_ms;
    align_ms += other.align_ms;
    visibility_ms += other.visibility_ms;
    target_builds += other.target_builds;
    target_preparations += other.target_preparations;
    alignments += other.alignments;
    visibility_evaluations += other.visibility_evaluations;
    candidates += other.candidates;
    cache_hits += other.cache_hits;
    cache_misses += other.cache_misses;
    effective_cache_max_bytes = other.effective_cache_max_bytes;
    effective_cache_max_entries = other.effective_cache_max_entries;
  }
};

} // namespace n3mapping
