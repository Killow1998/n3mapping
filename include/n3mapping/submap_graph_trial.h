// Isolated, no-writeback optimization trial for the shadow submap graph.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "n3mapping/config.h"
#include "n3mapping/submap_graph_projection.h"

namespace n3mapping {

struct SubmapGraphTrialNodeResult {
    SubmapId submap_id = kInvalidSubmapId;
    bool gauge_anchor = false;
    Eigen::Isometry3d initial_pose = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d optimized_pose = Eigen::Isometry3d::Identity();
    double translation_delta_m = 0.0;
    double rotation_delta_rad = 0.0;
};

struct SubmapGraphTrialPoseComparisonStats {
    std::size_t count = 0;
    double mean_translation_error_m = 0.0;
    double p95_translation_error_m = 0.0;
    double max_translation_error_m = 0.0;
    double mean_rotation_error_rad = 0.0;
    double p95_rotation_error_rad = 0.0;
    double max_rotation_error_rad = 0.0;
};

struct SubmapGraphTrialKeyframeResult {
    int64_t keyframe_id = -1;
    SubmapId submap_id = kInvalidSubmapId;
    Eigen::Isometry3d reference_pose = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d initial_shadow_pose = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d optimized_shadow_pose = Eigen::Isometry3d::Identity();
    double initial_translation_error_m = 0.0;
    double initial_rotation_error_rad = 0.0;
    double optimized_translation_error_m = 0.0;
    double optimized_rotation_error_rad = 0.0;
};

// A successful result is still diagnostic only. It carries no authority to
// update Submap, Keyframe, GraphOptimizer, map output, or runtime decisions.
struct SubmapGraphTrialDiagnostics {
    bool valid = false;
    bool attempted = false;
    bool solved = false;
    std::string failure_reason;
    std::size_t node_count = 0;
    std::size_t gauge_anchor_count = 0;
    std::size_t active_edge_factor_count = 0;
    std::size_t intra_submap_constant_edge_count = 0;
    std::size_t full_6d_factor_count = 0;
    std::size_t xy_yaw_lifted_factor_count = 0;
    std::size_t robust_factor_count = 0;
    std::size_t session_odometry_factor_count = 0;
    std::size_t explicit_information_factor_count = 0;
    std::size_t fallback_noise_factor_count = 0;
    std::size_t floor_factor_count = 0;
    double initial_nonlinear_error =
        std::numeric_limits<double>::quiet_NaN();
    double final_nonlinear_error =
        std::numeric_limits<double>::quiet_NaN();
    double nonlinear_error_reduction =
        std::numeric_limits<double>::quiet_NaN();
    double max_translation_delta_m = 0.0;
    double max_rotation_delta_rad = 0.0;
    SubmapGraphTrialPoseComparisonStats initial_keyframe_comparison;
    SubmapGraphTrialPoseComparisonStats optimized_keyframe_comparison;
    std::vector<SubmapGraphTrialNodeResult> nodes;
    std::vector<SubmapGraphTrialKeyframeResult> keyframes;
};

// Builds and solves a temporary batch graph from an immutable SG-02 snapshot.
// SG-03 readiness and SG-04 factor semantics are re-evaluated internally.
// The graph is destroyed before return and no input object is mutated.
SubmapGraphTrialDiagnostics evaluateSubmapGraphOptimizationTrial(
    const SubmapGraphSnapshot& snapshot,
    const Config& config);

}  // namespace n3mapping
