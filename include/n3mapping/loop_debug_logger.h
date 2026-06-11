#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/loop_detector.h"

namespace n3mapping {

struct LoopDebugCandidateEvent {
    double processing_time = 0.0;
    double query_timestamp = 0.0;
    LoopCandidate candidate;
    bool icp_converged = false;
    double fitness_score = std::numeric_limits<double>::quiet_NaN();
    double inlier_ratio = std::numeric_limits<double>::quiet_NaN();
    double icp_translation_norm = std::numeric_limits<double>::quiet_NaN();
    double icp_rotation_norm = std::numeric_limits<double>::quiet_NaN();
    Eigen::Isometry3d residual = Eigen::Isometry3d::Identity();
    bool has_loop_measurement = false;
    Eigen::Isometry3d loop_measurement_match_query = Eigen::Isometry3d::Identity();
    std::string edge_mode = "not_applicable";
    double vertical_observability_score = std::numeric_limits<double>::quiet_NaN();
    bool vertical_downweighted = false;
    double source_z_span = std::numeric_limits<double>::quiet_NaN();
    double target_z_span = std::numeric_limits<double>::quiet_NaN();
    double z_overlap_ratio_before = std::numeric_limits<double>::quiet_NaN();
    double z_overlap_ratio_after = std::numeric_limits<double>::quiet_NaN();
    double source_z_robust_span = std::numeric_limits<double>::quiet_NaN();
    double target_z_robust_span = std::numeric_limits<double>::quiet_NaN();
    double z_robust_overlap_ratio_before = std::numeric_limits<double>::quiet_NaN();
    double z_robust_overlap_ratio_after = std::numeric_limits<double>::quiet_NaN();
    double source_target_z_centroid_delta_before = std::numeric_limits<double>::quiet_NaN();
    double source_target_z_centroid_delta_after = std::numeric_limits<double>::quiet_NaN();
    double vertical_information_ratio = std::numeric_limits<double>::quiet_NaN();
    std::string gate_result = "rejected";
    std::string reject_reason;
    Eigen::Matrix<double, 6, 6> loop_information = Eigen::Matrix<double, 6, 6>::Identity();
};

struct LoopDebugOptimizationEvent {
    double processing_time = 0.0;
    std::size_t accepted_edge_count = 0;
    std::vector<std::pair<int64_t, int64_t>> accepted_edges;
    double loop_residual_translation_before = 0.0;
    double loop_residual_translation_after = 0.0;
    double loop_residual_rotation_before = 0.0;
    double loop_residual_rotation_after = 0.0;
    Eigen::Vector3d loop_residual_translation_axes_before = Eigen::Vector3d::Zero();
    Eigen::Vector3d loop_residual_translation_axes_after = Eigen::Vector3d::Zero();
    Eigen::Vector3d loop_residual_rpy_axes_before = Eigen::Vector3d::Zero();
    Eigen::Vector3d loop_residual_rpy_axes_after = Eigen::Vector3d::Zero();
    double mean_pose_update_translation = 0.0;
    double max_pose_update_translation = 0.0;
    double mean_pose_update_rotation = 0.0;
    double max_pose_update_rotation = 0.0;
};

class LoopDebugLogger {
  public:
    static std::string resolvePath(const Config& config);
    static bool appendCandidate(const std::string& path, const LoopDebugCandidateEvent& event);
    static bool appendOptimizationSummary(const std::string& path, const LoopDebugOptimizationEvent& event);
};

}  // namespace n3mapping
