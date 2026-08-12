// Shadow-only factor semantics for projected global-submap graphs.
#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "n3mapping/submap_graph_projection.h"

namespace n3mapping {

// Evaluates the existing keyframe factor after substituting
// X_keyframe = X_submap * T_submap_keyframe. No optimizer state is created.
struct SubmapGraphEdgeFactorEvaluation {
    bool valid = false;
    std::string failure_reason;
    std::size_t edge_projection_index = 0;
    std::size_t source_edge_index = 0;
    SubmapId from_submap_id = kInvalidSubmapId;
    SubmapId to_submap_id = kInvalidSubmapId;
    EdgeConstraintMode constraint_mode = EdgeConstraintMode::FULL_6DOF;

    // Exact residual used by the reference factor. FULL_6DOF follows GTSAM
    // Pose3 tangent order [rotation, translation]; XY_YAW is [x, y, yaw].
    Eigen::VectorXd exact_lifted_residual;
    bool explicit_source_information_usable = false;
    std::string information_limitation;
    Eigen::MatrixXd source_information_factor_order;
    double exact_lifted_mahalanobis_squared =
        std::numeric_limits<double>::quiet_NaN();

    // A constant projected BetweenFactor is exactly equivalent only for the
    // FULL_6DOF exponential-map residual. XY_YAW must retain the lifted
    // keyframe residual instead of being approximated as a submap factor.
    bool direct_between_residual_equivalent = false;
    bool direct_between_information_transportable = false;
    std::string direct_between_limitation;
    Eigen::Matrix<double, 6, 1> direct_between_residual =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> direct_between_information =
        Eigen::Matrix<double, 6, 6>::Zero();
    double residual_transport_error_norm =
        std::numeric_limits<double>::quiet_NaN();
    double direct_between_mahalanobis_squared =
        std::numeric_limits<double>::quiet_NaN();
    double mahalanobis_squared_delta =
        std::numeric_limits<double>::quiet_NaN();
};

struct SubmapGraphFloorFactorEvaluation {
    bool valid = false;
    std::string failure_reason;
    std::size_t floor_projection_index = 0;
    std::size_t source_constraint_index = 0;
    SubmapId submap_id = kInvalidSubmapId;
    Eigen::Vector3d normal_submap = Eigen::Vector3d::UnitZ();
    Eigen::Vector2d exact_lifted_residual = Eigen::Vector2d::Zero();
    Eigen::Vector2d projected_submap_residual = Eigen::Vector2d::Zero();
    double residual_error_norm =
        std::numeric_limits<double>::quiet_NaN();
    double exact_lifted_mahalanobis_squared =
        std::numeric_limits<double>::quiet_NaN();
    double projected_submap_mahalanobis_squared =
        std::numeric_limits<double>::quiet_NaN();
};

struct SubmapGraphFactorDiagnostics {
    bool valid = false;
    std::string failure_reason;
    std::size_t cross_edge_count = 0;
    std::size_t full_6d_edge_count = 0;
    std::size_t xy_yaw_exact_lifted_only_count = 0;
    std::size_t direct_between_residual_equivalent_count = 0;
    std::size_t direct_between_information_transportable_count = 0;
    std::size_t information_fallback_required_count = 0;
    std::size_t assigned_floor_factor_count = 0;
    std::size_t unassigned_floor_factor_count = 0;
    double max_residual_transport_error_norm = 0.0;
    double max_mahalanobis_squared_delta = 0.0;
    double max_floor_residual_error_norm = 0.0;
    std::vector<SubmapGraphEdgeFactorEvaluation> edge_evaluations;
    std::vector<SubmapGraphFloorFactorEvaluation> floor_evaluations;
};

SubmapGraphEdgeFactorEvaluation evaluateSubmapGraphEdgeFactor(
    const SubmapGraphEdgeProjection& projection,
    const Eigen::Isometry3d& T_map_from_submap,
    const Eigen::Isometry3d& T_map_to_submap,
    std::size_t edge_projection_index = 0);

SubmapGraphFloorFactorEvaluation evaluateSubmapGraphFloorFactor(
    const SubmapGraphFloorProjection& projection,
    const Eigen::Isometry3d& T_map_submap,
    std::size_t floor_projection_index = 0);

SubmapGraphFactorDiagnostics evaluateSubmapGraphFactors(
    const SubmapGraphSnapshot& snapshot);

}  // namespace n3mapping
