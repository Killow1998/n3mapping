#include "n3mapping/submap_graph_factor.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

#include <Eigen/Cholesky>
#include <gtsam/config.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace n3mapping {
namespace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;

bool finiteRigidPose(const Eigen::Isometry3d& pose,
                     double tolerance = 1e-6) {
    if (!pose.matrix().allFinite()) return false;
    const Eigen::Matrix3d rotation = pose.linear();
    return (rotation.transpose() * rotation)
               .isApprox(Eigen::Matrix3d::Identity(), tolerance) &&
           std::abs(rotation.determinant() - 1.0) <= tolerance &&
           pose.matrix().row(3).isApprox(
               Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), tolerance);
}

gtsam::Pose3 toGtsam(const Eigen::Isometry3d& pose) {
    return gtsam::Pose3(gtsam::Rot3(pose.rotation()),
                        gtsam::Point3(pose.translation()));
}

Matrix6d informationToGtsamOrder(const Matrix6d& information) {
    Matrix6d reordered;
    reordered.block<3, 3>(0, 0) = information.block<3, 3>(3, 3);
    reordered.block<3, 3>(0, 3) = information.block<3, 3>(3, 0);
    reordered.block<3, 3>(3, 0) = information.block<3, 3>(0, 3);
    reordered.block<3, 3>(3, 3) = information.block<3, 3>(0, 0);
    return reordered;
}

bool positiveDefiniteInformation(const Matrix6d& information,
                                 Matrix6d* symmetric) {
    if (!symmetric || !information.allFinite() ||
        information.isZero(1e-10)) {
        return false;
    }
    const double symmetry_tolerance =
        1e-9 * std::max(1.0, information.norm());
    if ((information - information.transpose()).norm() >
        symmetry_tolerance) {
        return false;
    }
    *symmetric = 0.5 * (information + information.transpose());
    return Eigen::LLT<Matrix6d>(*symmetric).info() == Eigen::Success;
}

double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double squaredMahalanobis(const Eigen::VectorXd& residual,
                          const Eigen::MatrixXd& information) {
    return residual.dot(information * residual);
}

SubmapGraphEdgeFactorEvaluation invalidEdgeEvaluation(
    const std::string& reason,
    std::size_t edge_projection_index) {
    SubmapGraphEdgeFactorEvaluation evaluation;
    evaluation.edge_projection_index = edge_projection_index;
    evaluation.failure_reason = reason;
    return evaluation;
}

SubmapGraphFloorFactorEvaluation invalidFloorEvaluation(
    const std::string& reason,
    std::size_t floor_projection_index) {
    SubmapGraphFloorFactorEvaluation evaluation;
    evaluation.floor_projection_index = floor_projection_index;
    evaluation.failure_reason = reason;
    return evaluation;
}

SubmapGraphFactorDiagnostics invalidDiagnostics(const std::string& reason) {
    SubmapGraphFactorDiagnostics diagnostics;
    diagnostics.failure_reason = reason;
    return diagnostics;
}

}  // namespace

SubmapGraphEdgeFactorEvaluation evaluateSubmapGraphEdgeFactor(
    const SubmapGraphEdgeProjection& projection,
    const Eigen::Isometry3d& T_map_from_submap,
    const Eigen::Isometry3d& T_map_to_submap,
    std::size_t edge_projection_index) {
    if (projection.classification !=
            SubmapGraphEdgeClass::CROSS_SUBMAP ||
        projection.from_submap_id == kInvalidSubmapId ||
        projection.to_submap_id == kInvalidSubmapId ||
        projection.from_submap_id == projection.to_submap_id ||
        projection.source_edge.from_id < 0 ||
        projection.source_edge.to_id < 0 ||
        projection.source_edge.from_id == projection.source_edge.to_id ||
        !projection.source_edge.information.allFinite() ||
        !projection.has_projected_measurement ||
        !finiteRigidPose(projection.source_edge.measurement) ||
        !finiteRigidPose(projection.T_from_submap_from_keyframe) ||
        !finiteRigidPose(projection.T_to_submap_to_keyframe) ||
        !finiteRigidPose(
            projection.T_from_submap_to_submap_measurement) ||
        !finiteRigidPose(T_map_from_submap) ||
        !finiteRigidPose(T_map_to_submap)) {
        return invalidEdgeEvaluation("invalid_cross_edge_factor_input",
                                     edge_projection_index);
    }

    SubmapGraphEdgeFactorEvaluation evaluation;
    evaluation.edge_projection_index = edge_projection_index;
    evaluation.source_edge_index = projection.source_edge_index;
    evaluation.from_submap_id = projection.from_submap_id;
    evaluation.to_submap_id = projection.to_submap_id;
    evaluation.constraint_mode = projection.source_edge.constraint_mode;

    const gtsam::Pose3 from_keyframe = toGtsam(
        T_map_from_submap *
        projection.T_from_submap_from_keyframe);
    const gtsam::Pose3 to_keyframe = toGtsam(
        T_map_to_submap * projection.T_to_submap_to_keyframe);
    const gtsam::Pose3 source_measurement =
        toGtsam(projection.source_edge.measurement);
    const gtsam::Pose3 source_prediction =
        from_keyframe.between(to_keyframe);

    switch (projection.source_edge.constraint_mode) {
        case EdgeConstraintMode::FULL_6DOF: {
            evaluation.exact_lifted_residual =
                source_measurement.localCoordinates(source_prediction);

            Matrix6d source_information;
            const Matrix6d reordered = informationToGtsamOrder(
                projection.source_edge.information);
            if (positiveDefiniteInformation(reordered,
                                            &source_information)) {
                evaluation.explicit_source_information_usable = true;
                evaluation.source_information_factor_order =
                    source_information;
                evaluation.exact_lifted_mahalanobis_squared =
                    squaredMahalanobis(
                        evaluation.exact_lifted_residual,
                        evaluation.source_information_factor_order);
            } else {
                evaluation.information_limitation =
                    "full_6d_config_fallback_or_invalid_information";
            }

#ifdef GTSAM_POSE3_EXPMAP
            const gtsam::Pose3 direct_measurement = toGtsam(
                projection.T_from_submap_to_submap_measurement);
            const gtsam::Pose3 direct_prediction =
                toGtsam(T_map_from_submap).between(
                    toGtsam(T_map_to_submap));
            evaluation.direct_between_residual =
                direct_measurement.localCoordinates(direct_prediction);
            evaluation.direct_between_residual_equivalent = true;

            const gtsam::Matrix6 adjoint = toGtsam(
                projection.T_to_submap_to_keyframe).AdjointMap();
            evaluation.residual_transport_error_norm =
                (evaluation.direct_between_residual -
                 adjoint * evaluation.exact_lifted_residual).norm();

            if (evaluation.explicit_source_information_usable) {
                const gtsam::Matrix6 adjoint_inverse = toGtsam(
                    projection.T_to_submap_to_keyframe)
                        .inverse().AdjointMap();
                const Matrix6d transported_information =
                    adjoint_inverse.transpose() * source_information *
                    adjoint_inverse;
                Matrix6d validated_information;
                if (positiveDefiniteInformation(
                        transported_information,
                        &validated_information)) {
                    evaluation.direct_between_information =
                        validated_information;
                    evaluation
                        .direct_between_information_transportable = true;
                    evaluation.direct_between_mahalanobis_squared =
                        evaluation.direct_between_residual.dot(
                            evaluation.direct_between_information *
                            evaluation.direct_between_residual);
                    evaluation.mahalanobis_squared_delta = std::abs(
                        evaluation.direct_between_mahalanobis_squared -
                        evaluation.exact_lifted_mahalanobis_squared);
                } else {
                    evaluation.direct_between_limitation =
                        "transported_information_not_positive_definite";
                }
            } else {
                evaluation.direct_between_limitation =
                    "explicit_source_information_unavailable";
            }
#else
            evaluation.direct_between_limitation =
                "gtsam_pose3_expmap_disabled";
#endif
            break;
        }
        case EdgeConstraintMode::XY_YAW: {
            evaluation.exact_lifted_residual.resize(3);
            evaluation.exact_lifted_residual <<
                source_prediction.translation().x() -
                    source_measurement.translation().x(),
                source_prediction.translation().y() -
                    source_measurement.translation().y(),
                normalizeAngle(source_prediction.rotation().yaw() -
                               source_measurement.rotation().yaw());

            const auto& information = projection.source_edge.information;
            if (information(0, 0) > 1e-12 &&
                information(1, 1) > 1e-12 &&
                information(5, 5) > 1e-12) {
                evaluation.explicit_source_information_usable = true;
                evaluation.source_information_factor_order =
                    Eigen::Matrix3d::Zero();
                evaluation.source_information_factor_order(0, 0) =
                    information(0, 0);
                evaluation.source_information_factor_order(1, 1) =
                    information(1, 1);
                evaluation.source_information_factor_order(2, 2) =
                    information(5, 5);
                evaluation.exact_lifted_mahalanobis_squared =
                    squaredMahalanobis(
                        evaluation.exact_lifted_residual,
                        evaluation.source_information_factor_order);
            } else {
                evaluation.information_limitation =
                    "xy_yaw_config_fallback_required";
            }
            evaluation.direct_between_limitation =
                "xy_yaw_requires_exact_lifted_factor";
            break;
        }
        default:
            return invalidEdgeEvaluation("invalid_edge_constraint_mode",
                                         edge_projection_index);
    }

    if (!evaluation.exact_lifted_residual.allFinite() ||
        (evaluation.explicit_source_information_usable &&
         (!evaluation.source_information_factor_order.allFinite() ||
          !std::isfinite(
              evaluation.exact_lifted_mahalanobis_squared))) ||
        (evaluation.direct_between_residual_equivalent &&
         (!evaluation.direct_between_residual.allFinite() ||
          !std::isfinite(evaluation.residual_transport_error_norm))) ||
        (evaluation.direct_between_information_transportable &&
         (!evaluation.direct_between_information.allFinite() ||
          !std::isfinite(
              evaluation.direct_between_mahalanobis_squared) ||
          !std::isfinite(evaluation.mahalanobis_squared_delta)))) {
        return invalidEdgeEvaluation("nonfinite_edge_factor_evaluation",
                                     edge_projection_index);
    }
    evaluation.valid = true;
    return evaluation;
}

SubmapGraphFloorFactorEvaluation evaluateSubmapGraphFloorFactor(
    const SubmapGraphFloorProjection& projection,
    const Eigen::Isometry3d& T_map_submap,
    std::size_t floor_projection_index) {
    const auto& constraint = projection.source_constraint;
    if (!projection.assigned ||
        projection.submap_id == kInvalidSubmapId ||
        constraint.node_id < 0 ||
        !constraint.normal_body.allFinite() ||
        constraint.normal_body.norm() < 1e-6 ||
        !std::isfinite(constraint.sigma_rad) ||
        constraint.sigma_rad <= 0.0 ||
        !finiteRigidPose(projection.T_submap_keyframe) ||
        !finiteRigidPose(T_map_submap)) {
        return invalidFloorEvaluation("invalid_floor_factor_input",
                                      floor_projection_index);
    }

    SubmapGraphFloorFactorEvaluation evaluation;
    evaluation.floor_projection_index = floor_projection_index;
    evaluation.source_constraint_index =
        projection.source_constraint_index;
    evaluation.submap_id = projection.submap_id;
    evaluation.normal_submap =
        projection.T_submap_keyframe.rotation() *
        constraint.normal_body.normalized();
    if (!evaluation.normal_submap.allFinite() ||
        evaluation.normal_submap.norm() < 1e-6) {
        return invalidFloorEvaluation("invalid_projected_floor_normal",
                                      floor_projection_index);
    }
    evaluation.normal_submap.normalize();

    const auto noise = gtsam::noiseModel::Isotropic::Sigma(
        2, constraint.sigma_rad);
    const gtsam::Pose3AttitudeFactor source_factor(
        0, gtsam::Unit3(0.0, 0.0, 1.0), noise,
        gtsam::Unit3(constraint.normal_body.normalized()));
    const gtsam::Pose3AttitudeFactor projected_factor(
        0, gtsam::Unit3(0.0, 0.0, 1.0), noise,
        gtsam::Unit3(evaluation.normal_submap));
    evaluation.exact_lifted_residual = source_factor.evaluateError(
        toGtsam(T_map_submap * projection.T_submap_keyframe));
    evaluation.projected_submap_residual = projected_factor.evaluateError(
        toGtsam(T_map_submap));
    evaluation.residual_error_norm =
        (evaluation.exact_lifted_residual -
         evaluation.projected_submap_residual).norm();
    const double information =
        1.0 / (constraint.sigma_rad * constraint.sigma_rad);
    evaluation.exact_lifted_mahalanobis_squared =
        information * evaluation.exact_lifted_residual.squaredNorm();
    evaluation.projected_submap_mahalanobis_squared =
        information * evaluation.projected_submap_residual.squaredNorm();
    if (!evaluation.exact_lifted_residual.allFinite() ||
        !evaluation.projected_submap_residual.allFinite() ||
        !std::isfinite(evaluation.residual_error_norm) ||
        !std::isfinite(evaluation.exact_lifted_mahalanobis_squared) ||
        !std::isfinite(evaluation.projected_submap_mahalanobis_squared)) {
        return invalidFloorEvaluation("nonfinite_floor_factor_evaluation",
                                      floor_projection_index);
    }
    evaluation.valid = true;
    return evaluation;
}

SubmapGraphFactorDiagnostics evaluateSubmapGraphFactors(
    const SubmapGraphSnapshot& snapshot) {
    if (!snapshot.valid) {
        return invalidDiagnostics("invalid_snapshot");
    }

    std::map<SubmapId, Eigen::Isometry3d> pose_by_submap;
    for (const auto& node : snapshot.nodes) {
        if (node.submap_id == kInvalidSubmapId ||
            !finiteRigidPose(node.T_map_submap) ||
            !pose_by_submap.emplace(node.submap_id,
                                    node.T_map_submap).second) {
            return invalidDiagnostics("invalid_factor_node");
        }
    }

    SubmapGraphFactorDiagnostics diagnostics;
    diagnostics.edge_evaluations.reserve(
        snapshot.cross_submap_edge_count);
    for (std::size_t index = 0;
         index < snapshot.edge_projections.size(); ++index) {
        const auto& projection = snapshot.edge_projections[index];
        if (projection.source_edge_index != index) {
            return invalidDiagnostics("inconsistent_factor_edge_index");
        }
        if (projection.classification !=
            SubmapGraphEdgeClass::CROSS_SUBMAP) {
            continue;
        }
        const auto from_pose = pose_by_submap.find(
            projection.from_submap_id);
        const auto to_pose = pose_by_submap.find(
            projection.to_submap_id);
        if (from_pose == pose_by_submap.end() ||
            to_pose == pose_by_submap.end()) {
            return invalidDiagnostics("missing_factor_edge_node");
        }
        auto evaluation = evaluateSubmapGraphEdgeFactor(
            projection, from_pose->second, to_pose->second, index);
        if (!evaluation.valid) {
            return invalidDiagnostics(evaluation.failure_reason);
        }
        ++diagnostics.cross_edge_count;
        if (evaluation.constraint_mode == EdgeConstraintMode::FULL_6DOF) {
            ++diagnostics.full_6d_edge_count;
        } else {
            ++diagnostics.xy_yaw_exact_lifted_only_count;
        }
        if (evaluation.direct_between_residual_equivalent) {
            ++diagnostics.direct_between_residual_equivalent_count;
            diagnostics.max_residual_transport_error_norm = std::max(
                diagnostics.max_residual_transport_error_norm,
                evaluation.residual_transport_error_norm);
        }
        if (evaluation.direct_between_information_transportable) {
            ++diagnostics
                 .direct_between_information_transportable_count;
            diagnostics.max_mahalanobis_squared_delta = std::max(
                diagnostics.max_mahalanobis_squared_delta,
                evaluation.mahalanobis_squared_delta);
        }
        if (!evaluation.explicit_source_information_usable) {
            ++diagnostics.information_fallback_required_count;
        }
        diagnostics.edge_evaluations.push_back(std::move(evaluation));
    }
    if (diagnostics.cross_edge_count !=
        snapshot.cross_submap_edge_count) {
        return invalidDiagnostics("inconsistent_factor_edge_count");
    }

    diagnostics.floor_evaluations.reserve(
        snapshot.assigned_floor_constraint_count);
    for (std::size_t index = 0;
         index < snapshot.floor_projections.size(); ++index) {
        const auto& projection = snapshot.floor_projections[index];
        if (projection.source_constraint_index != index) {
            return invalidDiagnostics("inconsistent_factor_floor_index");
        }
        if (!projection.assigned) {
            ++diagnostics.unassigned_floor_factor_count;
            continue;
        }
        const auto pose = pose_by_submap.find(projection.submap_id);
        if (pose == pose_by_submap.end()) {
            return invalidDiagnostics("missing_factor_floor_node");
        }
        auto evaluation = evaluateSubmapGraphFloorFactor(
            projection, pose->second, index);
        if (!evaluation.valid) {
            return invalidDiagnostics(evaluation.failure_reason);
        }
        ++diagnostics.assigned_floor_factor_count;
        diagnostics.max_floor_residual_error_norm = std::max(
            diagnostics.max_floor_residual_error_norm,
            evaluation.residual_error_norm);
        diagnostics.floor_evaluations.push_back(std::move(evaluation));
    }
    if (diagnostics.assigned_floor_factor_count !=
            snapshot.assigned_floor_constraint_count ||
        diagnostics.unassigned_floor_factor_count !=
            snapshot.unassigned_floor_constraint_count) {
        return invalidDiagnostics("inconsistent_factor_floor_count");
    }

    diagnostics.valid = true;
    return diagnostics;
}

}  // namespace n3mapping
