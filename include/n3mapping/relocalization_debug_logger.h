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

namespace n3mapping {

struct RelocDebugBasinSummary {
    int64_t center_match_id = -1;
    std::vector<int64_t> member_match_ids;
};

struct RelocDebugBasinBestSummary {
    int64_t basin_center_id = -1;
    int64_t matched_kf_id = -1;
    LoopCandidate candidate;
    double fitness_score = std::numeric_limits<double>::quiet_NaN();
    double inlier_ratio = std::numeric_limits<double>::quiet_NaN();
    double selection_score = std::numeric_limits<double>::quiet_NaN();
    double log_likelihood = std::numeric_limits<double>::quiet_NaN();
};

struct RelocDebugHypothesisSummary {
    int64_t seed_match_id = -1;
    int64_t last_match_id = -1;
    double cumulative_log_likelihood = std::numeric_limits<double>::quiet_NaN();
    int num_updates = 0;
    int converged_updates = 0;
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
    uint64_t query_index = 0;
    RelocQueryCloudDebugSummary query_cloud;
    RelocQueryCloudDebugSummary motion_query_cloud;
    std::size_t candidate_count = 0;
    std::vector<LoopCandidate> top_candidates;
    std::vector<RelocDebugBasinSummary> basins;
    std::vector<RelocDebugBasinBestSummary> basin_best_results;
    std::vector<RelocDebugHypothesisSummary> hypotheses;
    double temporal_hypothesis_score = std::numeric_limits<double>::quiet_NaN();
    double log_likelihood = std::numeric_limits<double>::quiet_NaN();
    int winner_streak = 0;
    double margin = std::numeric_limits<double>::quiet_NaN();
    double ratio = std::numeric_limits<double>::quiet_NaN();
    double basin_separation = std::numeric_limits<double>::quiet_NaN();
    bool lock_accepted = false;
    std::string lock_result = "rejected";
    std::string reject_reason;
};

struct RelocTrackingDebugEvent {
    double processing_time = 0.0;
    uint64_t query_index = 0;
    Eigen::Isometry3d predicted_pose = Eigen::Isometry3d::Identity();
    int64_t nearest_kf_id = -1;
    std::size_t submap_size = 0;
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
    static std::string resolvePath(const Config& config);
    static bool appendRelocalization(const std::string& path, const RelocalizationDebugEvent& event);
    static bool appendTracking(const std::string& path, const RelocTrackingDebugEvent& event);
};

}  // namespace n3mapping
