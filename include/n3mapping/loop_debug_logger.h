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
    size_t icp_iterations = 0;
    double icp_optimizer_error = std::numeric_limits<double>::quiet_NaN();
    std::string icp_termination = "invalid";
    double fitness_score = std::numeric_limits<double>::quiet_NaN();
    double inlier_ratio = std::numeric_limits<double>::quiet_NaN();
    double icp_translation_norm = std::numeric_limits<double>::quiet_NaN();
    double icp_rotation_norm = std::numeric_limits<double>::quiet_NaN();
    Eigen::Isometry3d residual = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_pred_match_query = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_icp_correction_match = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_measured_match_query = Eigen::Isometry3d::Identity();
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
    int vertical_hypothesis_count = 0;
    double best_z_offset_m = std::numeric_limits<double>::quiet_NaN();
    double best_z_offset_fitness = std::numeric_limits<double>::quiet_NaN();
    double zero_z_fitness = std::numeric_limits<double>::quiet_NaN();
    double fitness_gap_zero_vs_best = std::numeric_limits<double>::quiet_NaN();
    double z_hypothesis_spread_m = std::numeric_limits<double>::quiet_NaN();
    double vertical_ambiguity_score = std::numeric_limits<double>::quiet_NaN();
    std::string vertical_hypothesis_edge_recommendation = "not_available";
    int heightmap_overlap_cell_count = 0;
    double heightmap_overlap_ratio = 0.0;
    double heightmap_ground_dz_median = std::numeric_limits<double>::quiet_NaN();
    double heightmap_ground_dz_p90 = std::numeric_limits<double>::quiet_NaN();
    double heightmap_ground_dz_max = std::numeric_limits<double>::quiet_NaN();
    double heightmap_ground_support_ratio = 0.0;
    double heightmap_vertical_consistency_score = 0.0;
    bool graph_trial_success = false;
    double graph_trial_residual_x_after = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_residual_y_after = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_residual_z_after = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_residual_roll_after = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_residual_pitch_after = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_residual_yaw_after = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_residual_translation_norm_after = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_residual_rotation_norm_after = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_mean_pose_update_translation = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_max_pose_update_translation = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_mean_pose_update_rotation = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_max_pose_update_rotation = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_existing_loop_residual_delta = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_odom_residual_delta = std::numeric_limits<double>::quiet_NaN();
    double graph_trial_consistency_score = std::numeric_limits<double>::quiet_NaN();
    std::string graph_trial_recommendation = "not_available";
    int segment_pair_count = 0;
    int segment_valid_pair_count = 0;
    int segment_consensus_inlier_count = 0;
    double segment_consensus_ratio = 0.0;
    double segment_translation_median = std::numeric_limits<double>::quiet_NaN();
    double segment_translation_std = std::numeric_limits<double>::quiet_NaN();
    double segment_yaw_median = std::numeric_limits<double>::quiet_NaN();
    double segment_yaw_std = std::numeric_limits<double>::quiet_NaN();
    double segment_z_std = std::numeric_limits<double>::quiet_NaN();
    double segment_roll_pitch_std = std::numeric_limits<double>::quiet_NaN();
    std::string segment_direction = "not_available";
    std::string segment_recommendation = "not_available";
    std::string consensus_shadow_decision = "not_available";
    std::string consensus_shadow_reason = "not_available";
    int consensus_valid_pair_count = 0;
    int consensus_left_support_count = 0;
    int consensus_right_support_count = 0;
    int consensus_contradiction_count = 0;
    double consensus_median_translation_delta = std::numeric_limits<double>::quiet_NaN();
    double consensus_mad_translation_delta = std::numeric_limits<double>::quiet_NaN();
    double consensus_median_rotation_delta = std::numeric_limits<double>::quiet_NaN();
    double consensus_mad_rotation_delta = std::numeric_limits<double>::quiet_NaN();
    bool consensus_estimator_valid = false;
    int consensus_estimator_pair_count = 0;
    int consensus_estimator_inlier_count = 0;
    double consensus_estimator_inlier_ratio = 0.0;
    double consensus_estimator_translation_median = std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_z_median = std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_yaw_median = std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_translation_mad = std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_z_mad = std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_yaw_mad = std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_measurement_delta_translation =
        std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_measurement_delta_rotation =
        std::numeric_limits<double>::quiet_NaN();
    std::string consensus_estimator_recommendation = "not_available";
    bool consensus_estimator_trial_success = false;
    double consensus_estimator_trial_residual_x_after =
        std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_trial_residual_y_after =
        std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_trial_residual_z_after =
        std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_trial_residual_roll_after =
        std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_trial_residual_pitch_after =
        std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_trial_residual_yaw_after =
        std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_trial_residual_translation_norm_after =
        std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_trial_residual_rotation_norm_after =
        std::numeric_limits<double>::quiet_NaN();
    double consensus_estimator_trial_consistency_score =
        std::numeric_limits<double>::quiet_NaN();
    std::string consensus_estimator_trial_recommendation = "not_available";
    std::string loop_referee_recommendation = "not_available";
    std::string loop_referee_reason = "not_available";
    std::string loop_referee_risk_flags = "not_available";
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
