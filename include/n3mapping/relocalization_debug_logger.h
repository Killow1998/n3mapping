#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "n3mapping/config.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/registration_observability.h"

namespace n3mapping {

struct RelocDebugBasinSummary {
  int64_t center_match_id = -1;
  std::vector<int64_t> member_match_ids;
};

struct RelocDebugBasinBestSummary {
  int64_t basin_center_id = -1;
  int64_t matched_kf_id = -1;
  LoopCandidate candidate;
  Eigen::Isometry3d pose_in_map = Eigen::Isometry3d::Identity();
  double fitness_score = std::numeric_limits<double>::quiet_NaN();
  double inlier_ratio = std::numeric_limits<double>::quiet_NaN();
  double selection_score = std::numeric_limits<double>::quiet_NaN();
  double log_likelihood = std::numeric_limits<double>::quiet_NaN();
  double visibility_consistency_ratio =
      std::numeric_limits<double>::quiet_NaN();
  double visibility_observed_coverage =
      std::numeric_limits<double>::quiet_NaN();
  double visibility_foreground_conflict_ratio =
      std::numeric_limits<double>::quiet_NaN();
  double visibility_evidence_log_odds =
      std::numeric_limits<double>::quiet_NaN();
  std::size_t visibility_known_bins = 0;
  std::size_t visibility_unknown_bins = 0;
  double visibility_known_fraction =
      std::numeric_limits<double>::quiet_NaN();
  double visibility_consistent_given_known =
      std::numeric_limits<double>::quiet_NaN();
  double visibility_foreground_conflict_given_known =
      std::numeric_limits<double>::quiet_NaN();
  double visibility_evidence_log_odds_given_known =
      std::numeric_limits<double>::quiet_NaN();
  RegistrationObservability registration_observability;
};

struct RelocDebugHypothesisSummary {
  int64_t seed_match_id = -1;
  int64_t last_match_id = -1;
  Eigen::Isometry3d pose_in_map = Eigen::Isometry3d::Identity();
  double cumulative_log_likelihood = std::numeric_limits<double>::quiet_NaN();
  int num_updates = 0;
  int converged_updates = 0;
  int visibility_updates = 0;
  double mean_visibility_consistency = std::numeric_limits<double>::quiet_NaN();
  double mean_visibility_evidence = std::numeric_limits<double>::quiet_NaN();
  RegistrationObservability registration_observability;
  bool alive = false;
};

struct RelocQueryCloudDebugSummary {
  std::string mode;
  int frame_count = 0;
  double motion_translation_m = std::numeric_limits<double>::quiet_NaN();
  double motion_rotation_rad = std::numeric_limits<double>::quiet_NaN();
  std::size_t raw_points = 0;
  std::size_t downsampled_points = 0;
  std::size_t candidate_count = 0;
  std::vector<LoopCandidate> top_candidates;
};

struct RelocalizationDebugEvent {
  double processing_time = 0.0;
  double query_timestamp = std::numeric_limits<double>::quiet_NaN();
  uint64_t query_index = 0;
  RelocQueryCloudDebugSummary query_cloud;
  RelocQueryCloudDebugSummary motion_query_cloud;
  std::size_t candidate_count = 0;
  std::vector<LoopCandidate> top_candidates;
  std::vector<RelocDebugBasinSummary> basins;
  std::vector<RelocDebugBasinBestSummary> basin_best_results;
  std::vector<RelocDebugHypothesisSummary> hypotheses;
  int64_t winner_seed_match_id = -1;
  int64_t winner_last_match_id = -1;
  int64_t runner_up_seed_match_id = -1;
  int64_t runner_up_last_match_id = -1;
  double temporal_hypothesis_score = std::numeric_limits<double>::quiet_NaN();
  double log_likelihood = std::numeric_limits<double>::quiet_NaN();
  int winner_streak = 0;
  double winner_pose_translation_delta =
      std::numeric_limits<double>::quiet_NaN();
  double winner_pose_rotation_delta =
      std::numeric_limits<double>::quiet_NaN();
  double evidence_motion_translation =
      std::numeric_limits<double>::quiet_NaN();
  double evidence_motion_rotation =
      std::numeric_limits<double>::quiet_NaN();
  bool moving_visibility_required = false;
  bool moving_visibility_passed = true;
  double margin = std::numeric_limits<double>::quiet_NaN();
  double ratio = std::numeric_limits<double>::quiet_NaN();
  double visibility_margin = std::numeric_limits<double>::quiet_NaN();
  double visibility_ratio = std::numeric_limits<double>::quiet_NaN();
  double basin_separation = std::numeric_limits<double>::quiet_NaN();
  bool lock_accepted = false;
  std::string lock_result = "rejected";
  std::string reject_reason;
};

struct RelocTrackingDebugEvent {
  double processing_time = 0.0;
  uint64_t query_index = 0;
  bool strict_loaded_map = false;
  Eigen::Isometry3d predicted_pose = Eigen::Isometry3d::Identity();
  int64_t nearest_kf_id = -1;
  std::size_t submap_size = 0;
  // PERF-ME-01A: append-only, diagnostic-only stage timings. Missing stages
  // remain NaN and are serialized as JSON null, so an early return cannot be
  // mistaken for a zero-cost stage.
  double tracking_total_ms = std::numeric_limits<double>::quiet_NaN();
  double nearest_keyframe_ms = std::numeric_limits<double>::quiet_NaN();
  double loaded_map_cache_ms = std::numeric_limits<double>::quiet_NaN();
  double submap_build_ms = std::numeric_limits<double>::quiet_NaN();
  double target_prepare_ms = std::numeric_limits<double>::quiet_NaN();
  double source_prepare_ms = std::numeric_limits<double>::quiet_NaN();
  double registration_ms = std::numeric_limits<double>::quiet_NaN();
  double retry_registration_ms = std::numeric_limits<double>::quiet_NaN();
  double visibility_ms = std::numeric_limits<double>::quiet_NaN();
  bool icp_converged = false;
  double fitness_score = std::numeric_limits<double>::quiet_NaN();
  double inlier_ratio = std::numeric_limits<double>::quiet_NaN();
  bool retry_used = false;
  int consecutive_track_failures = 0;
  bool result_success = false;
  std::string reject_reason;
};

class RelocalizationDebugLogger {
public:
  static std::string resolvePath(const Config &config);
  static bool appendRelocalization(const std::string &path,
                                   const RelocalizationDebugEvent &event);
  static bool appendTracking(const std::string &path,
                             const RelocTrackingDebugEvent &event);
};

} // namespace n3mapping
