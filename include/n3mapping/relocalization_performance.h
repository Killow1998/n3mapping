#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>
#include "n3mapping/descriptor_performance.h"

namespace Json { class Value; }

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

// Per-observation costs and diagnostic snapshots. Timing never influences a
// decision. Zero time means the stage did not run; available distinguishes a
// search attempt from tracking.
struct RelocalizationPerformance {
  bool available = false;
  double total_ms = 0.0;
  // Process CPU includes all workers/background work in this interval.
  double process_cpu_ms = 0.0;
  double query_build_ms = 0.0;
  double free_space_ms = 0.0;
  double directional_check_ms = 0.0;
  bool directional_checked = false;
  // Copy of the lock witness for normal status/logs, not its decision state.
  double directional_min_support = std::numeric_limits<double>::quiet_NaN();
  double diagnostics_ms = 0.0;
  double fixed_pose_ms = 0.0;
  std::uint64_t fixed_pose_evaluations = 0;
  std::uint64_t rejected_alignments = 0;
  std::uint64_t seed_continuations = 0;
  std::uint64_t seed_resumptions = 0;
  std::uint64_t seed_initializations = 0;
  std::uint64_t seed_retries = 0;
  std::uint64_t invalidated_seed_poses = 0;
  std::uint64_t confirmation_evaluations = 0;
  std::uint64_t confirmation_registrations = 0;
  DescriptorSearchPerformance descriptor;
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
    fixed_pose_ms += other.fixed_pose_ms;
    fixed_pose_evaluations += other.fixed_pose_evaluations;
    rejected_alignments += other.rejected_alignments;
    seed_continuations += other.seed_continuations;
    seed_resumptions += other.seed_resumptions;
    seed_initializations += other.seed_initializations;
    seed_retries += other.seed_retries;
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

Json::Value relocalizationPerformanceJson(const RelocalizationPerformance &cost);

} // namespace n3mapping
