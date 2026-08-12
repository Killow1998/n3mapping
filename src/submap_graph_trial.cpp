#include "n3mapping/submap_graph_trial.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <utility>

#include <boost/make_shared.hpp>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <tbb/global_control.h>

#include "n3mapping/graph_factor_noise.h"
#include "n3mapping/submap_graph_factor.h"

namespace n3mapping {
namespace {

bool finiteRigidPose(const Eigen::Isometry3d& pose,
                     double tolerance = 1e-6) {
    if (!pose.matrix().allFinite()) return false;
    const Eigen::Matrix3d rotation = pose.rotation();
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

Eigen::Isometry3d toEigen(const gtsam::Pose3& pose) {
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.linear() = pose.rotation().matrix();
    result.translation() = pose.translation();
    return result;
}

double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double rotationAngle(const Eigen::Matrix3d& rotation) {
    return Eigen::AngleAxisd(rotation).angle();
}

class ExactLiftedSubmapEdgeFactor final
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
  public:
    using Base =
        gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;

    ExactLiftedSubmapEdgeFactor(
        gtsam::Key from_key,
        gtsam::Key to_key,
        const gtsam::Pose3& T_from_submap_from_keyframe,
        const gtsam::Pose3& T_to_submap_to_keyframe,
        const gtsam::Pose3& source_measurement,
        EdgeConstraintMode constraint_mode,
        const gtsam::SharedNoiseModel& noise)
      : Base(noise, from_key, to_key),
        T_from_submap_from_keyframe_(T_from_submap_from_keyframe),
        T_to_submap_to_keyframe_(T_to_submap_to_keyframe),
        source_measurement_(source_measurement),
        constraint_mode_(constraint_mode) {}

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(
                new ExactLiftedSubmapEdgeFactor(*this)));
    }

    gtsam::Vector evaluateError(
        const gtsam::Pose3& from_submap,
        const gtsam::Pose3& to_submap,
        boost::optional<gtsam::Matrix&> H1 = boost::none,
        boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        if (H1) {
            *H1 = numericalJacobian(
                [&](const gtsam::Pose3& pose) {
                    return errorVector(pose, to_submap);
                }, from_submap);
        }
        if (H2) {
            *H2 = numericalJacobian(
                [&](const gtsam::Pose3& pose) {
                    return errorVector(from_submap, pose);
                }, to_submap);
        }
        return errorVector(from_submap, to_submap);
    }

  private:
    gtsam::Pose3 T_from_submap_from_keyframe_;
    gtsam::Pose3 T_to_submap_to_keyframe_;
    gtsam::Pose3 source_measurement_;
    EdgeConstraintMode constraint_mode_;

    gtsam::Vector errorVector(const gtsam::Pose3& from_submap,
                              const gtsam::Pose3& to_submap) const {
        const gtsam::Pose3 source_prediction =
            from_submap.compose(T_from_submap_from_keyframe_)
                .between(to_submap.compose(
                    T_to_submap_to_keyframe_));
        if (constraint_mode_ == EdgeConstraintMode::FULL_6DOF) {
            return source_measurement_.localCoordinates(source_prediction);
        }
        Eigen::Vector3d error;
        error << source_prediction.translation().x() -
                     source_measurement_.translation().x(),
            source_prediction.translation().y() -
                source_measurement_.translation().y(),
            normalizeAngle(source_prediction.rotation().yaw() -
                           source_measurement_.rotation().yaw());
        return error;
    }

    gtsam::Matrix numericalJacobian(
        const std::function<gtsam::Vector(const gtsam::Pose3&)>& function,
        const gtsam::Pose3& pose) const {
        constexpr double kDelta = 1e-5;
        const std::size_t dimension =
            constraint_mode_ == EdgeConstraintMode::FULL_6DOF ? 6 : 3;
        gtsam::Matrix jacobian = gtsam::Matrix::Zero(dimension, 6);
        for (int column = 0; column < 6; ++column) {
            gtsam::Vector6 delta = gtsam::Vector6::Zero();
            delta(column) = kDelta;
            const gtsam::Vector plus = function(pose.retract(delta));
            delta(column) = -kDelta;
            const gtsam::Vector minus = function(pose.retract(delta));
            jacobian.col(column) = (plus - minus) / (2.0 * kDelta);
        }
        return jacobian;
    }
};

SubmapGraphTrialDiagnostics invalidTrial(const std::string& reason,
                                         bool attempted = false) {
    SubmapGraphTrialDiagnostics diagnostics;
    diagnostics.attempted = attempted;
    diagnostics.failure_reason = reason;
    return diagnostics;
}

}  // namespace

SubmapGraphTrialDiagnostics evaluateSubmapGraphOptimizationTrial(
    const SubmapGraphSnapshot& snapshot,
    const Config& config) {
    std::string config_error;
    if (!config.validate(&config_error)) {
        return invalidTrial("invalid_config:" + config_error);
    }
    const auto topology = evaluateSubmapGraphTopology(snapshot);
    if (!topology.valid) {
        return invalidTrial("invalid_topology:" + topology.failure_reason);
    }
    if (!topology.shadow_graph_ready || topology.component_count != 1 ||
        topology.components.size() != 1) {
        return invalidTrial("topology_not_ready");
    }
    const auto semantics = evaluateSubmapGraphFactors(snapshot);
    if (!semantics.valid) {
        return invalidTrial("invalid_factor_semantics:" +
                            semantics.failure_reason);
    }

    SubmapGraphTrialDiagnostics diagnostics;
    diagnostics.attempted = true;
    diagnostics.node_count = snapshot.nodes.size();
    if (snapshot.nodes.empty()) {
        return invalidTrial("empty_trial_graph", true);
    }

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_values;
    std::map<SubmapId, gtsam::Key> key_by_submap;
    for (std::size_t index = 0; index < snapshot.nodes.size(); ++index) {
        const auto& node = snapshot.nodes[index];
        if (!finiteRigidPose(node.T_map_submap)) {
            return invalidTrial("invalid_trial_node_pose", true);
        }
        const gtsam::Key key = gtsam::Symbol('s', index);
        if (!key_by_submap.emplace(node.submap_id, key).second) {
            return invalidTrial("duplicate_trial_node", true);
        }
        initial_values.insert(key, toGtsam(node.T_map_submap));
    }

    const SubmapId gauge_anchor_id =
        topology.components.front().anchor_submap_id;
    const auto gauge_key = key_by_submap.find(gauge_anchor_id);
    if (gauge_key == key_by_submap.end()) {
        return invalidTrial("missing_gauge_anchor", true);
    }
    graph.add(gtsam::NonlinearEquality<gtsam::Pose3>(
        gauge_key->second,
        initial_values.at<gtsam::Pose3>(gauge_key->second)));
    diagnostics.gauge_anchor_count = 1;

    int64_t first_session_node_id = std::numeric_limits<int64_t>::max();
    for (const auto& projection : snapshot.edge_projections) {
        if (projection.source_edge.type == EdgeType::SESSION_ANCHOR) {
            first_session_node_id = std::min(
                first_session_node_id, projection.source_edge.to_id);
        }
    }

    for (const auto& projection : snapshot.edge_projections) {
        if (projection.classification ==
            SubmapGraphEdgeClass::INTRA_SUBMAP) {
            ++diagnostics.intra_submap_constant_edge_count;
            continue;
        }
        if (projection.classification !=
            SubmapGraphEdgeClass::CROSS_SUBMAP) {
            return invalidTrial("unassigned_trial_edge", true);
        }
        const auto from_key = key_by_submap.find(projection.from_submap_id);
        const auto to_key = key_by_submap.find(projection.to_submap_id);
        if (from_key == key_by_submap.end() ||
            to_key == key_by_submap.end()) {
            return invalidTrial("missing_trial_edge_node", true);
        }
        const EdgeInfo& edge = projection.source_edge;
        const bool session_odometry =
            edge.type == EdgeType::ODOMETRY &&
            first_session_node_id != std::numeric_limits<int64_t>::max() &&
            edge.from_id >= first_session_node_id &&
            edge.to_id >= first_session_node_id;
        const bool robust_requested =
            GraphOptimizer::isRobustGlobalEdge(edge.type) ||
            session_odometry;
        if (edge.constraint_mode == EdgeConstraintMode::XY_YAW &&
            edge.type == EdgeType::ODOMETRY) {
            return invalidTrial("unsupported_xy_yaw_odometry", true);
        }
        GraphFactorNoiseSelection noise;
        if (edge.constraint_mode == EdgeConstraintMode::FULL_6DOF) {
            noise = makeFullGraphFactorNoise(
                edge.information, config,
                robust_requested
                    ? GraphFactorNoiseRole::
                          ROBUST_GLOBAL_OR_SESSION_ODOMETRY
                    : GraphFactorNoiseRole::ODOMETRY,
                robust_requested);
            ++diagnostics.full_6d_factor_count;
        } else if (edge.constraint_mode == EdgeConstraintMode::XY_YAW) {
            noise = makeXYYawGraphFactorNoise(
                edge.information, config, robust_requested);
            ++diagnostics.xy_yaw_lifted_factor_count;
        } else {
            return invalidTrial("invalid_trial_constraint_mode", true);
        }
        if (!noise.model) {
            return invalidTrial("missing_trial_noise_model", true);
        }
        graph.add(boost::make_shared<ExactLiftedSubmapEdgeFactor>(
            from_key->second, to_key->second,
            toGtsam(projection.T_from_submap_from_keyframe),
            toGtsam(projection.T_to_submap_to_keyframe),
            toGtsam(edge.measurement), edge.constraint_mode,
            noise.model));
        ++diagnostics.active_edge_factor_count;
        if (session_odometry) {
            ++diagnostics.session_odometry_factor_count;
        }
        if (noise.robust) ++diagnostics.robust_factor_count;
        if (noise.explicit_information) {
            ++diagnostics.explicit_information_factor_count;
        }
        if (noise.fallback) ++diagnostics.fallback_noise_factor_count;
    }
    if (diagnostics.active_edge_factor_count !=
            snapshot.cross_submap_edge_count ||
        diagnostics.intra_submap_constant_edge_count !=
            snapshot.intra_submap_edge_count) {
        return invalidTrial("inconsistent_trial_edge_count", true);
    }

    for (const auto& projection : snapshot.floor_projections) {
        if (!projection.assigned) {
            return invalidTrial("unassigned_trial_floor", true);
        }
        const auto key = key_by_submap.find(projection.submap_id);
        if (key == key_by_submap.end()) {
            return invalidTrial("missing_trial_floor_node", true);
        }
        const auto floor = evaluateSubmapGraphFloorFactor(
            projection,
            toEigen(initial_values.at<gtsam::Pose3>(key->second)));
        if (!floor.valid) {
            return invalidTrial("invalid_trial_floor:" +
                                floor.failure_reason, true);
        }
        const auto noise = gtsam::noiseModel::Isotropic::Sigma(
            2, projection.source_constraint.sigma_rad);
        graph.add(gtsam::Pose3AttitudeFactor(
            key->second, gtsam::Unit3(0.0, 0.0, 1.0), noise,
            gtsam::Unit3(floor.normal_submap)));
        ++diagnostics.floor_factor_count;
    }
    if (diagnostics.floor_factor_count !=
        snapshot.assigned_floor_constraint_count) {
        return invalidTrial("inconsistent_trial_floor_count", true);
    }

    try {
        diagnostics.initial_nonlinear_error = graph.error(initial_values);
        gtsam::LevenbergMarquardtParams parameters;
        parameters.maxIterations = config.optimization_iterations;
        parameters.verbosity =
            gtsam::NonlinearOptimizerParams::SILENT;
        tbb::global_control serial_trial(
            tbb::global_control::max_allowed_parallelism, 1);
        const gtsam::Values optimized =
            gtsam::LevenbergMarquardtOptimizer(
                graph, initial_values, parameters).optimize();
        diagnostics.final_nonlinear_error = graph.error(optimized);
        if (!std::isfinite(diagnostics.initial_nonlinear_error) ||
            !std::isfinite(diagnostics.final_nonlinear_error) ||
            optimized.size() != initial_values.size()) {
            return invalidTrial("invalid_trial_solution", true);
        }
        const double error_tolerance = 1e-9 * std::max(
            1.0, diagnostics.initial_nonlinear_error);
        if (diagnostics.final_nonlinear_error >
            diagnostics.initial_nonlinear_error + error_tolerance) {
            return invalidTrial("trial_error_increased", true);
        }
        diagnostics.nonlinear_error_reduction =
            diagnostics.initial_nonlinear_error -
            diagnostics.final_nonlinear_error;
        diagnostics.nodes.reserve(snapshot.nodes.size());
        for (const auto& node : snapshot.nodes) {
            const gtsam::Key key = key_by_submap.at(node.submap_id);
            if (!optimized.exists(key)) {
                return invalidTrial("missing_trial_solution_node", true);
            }
            SubmapGraphTrialNodeResult result;
            result.submap_id = node.submap_id;
            result.gauge_anchor = node.submap_id == gauge_anchor_id;
            result.initial_pose = node.T_map_submap;
            result.optimized_pose = toEigen(
                optimized.at<gtsam::Pose3>(key));
            if (!finiteRigidPose(result.optimized_pose)) {
                return invalidTrial("nonrigid_trial_solution", true);
            }
            const Eigen::Isometry3d delta =
                result.initial_pose.inverse() * result.optimized_pose;
            result.translation_delta_m = delta.translation().norm();
            result.rotation_delta_rad = rotationAngle(delta.rotation());
            if (result.gauge_anchor &&
                !result.optimized_pose.matrix().isApprox(
                    result.initial_pose.matrix(), 1e-9)) {
                return invalidTrial("gauge_anchor_moved", true);
            }
            diagnostics.max_translation_delta_m = std::max(
                diagnostics.max_translation_delta_m,
                result.translation_delta_m);
            diagnostics.max_rotation_delta_rad = std::max(
                diagnostics.max_rotation_delta_rad,
                result.rotation_delta_rad);
            diagnostics.nodes.push_back(std::move(result));
        }
    } catch (const std::exception& error) {
        return invalidTrial("trial_optimization_exception:" +
                            std::string(error.what()), true);
    }

    diagnostics.solved = true;
    diagnostics.valid = true;
    return diagnostics;
}

}  // namespace n3mapping
