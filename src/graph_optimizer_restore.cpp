#include <glog/logging.h>
#include "n3mapping/graph_optimizer.h"
#include "graph_pose_factors.h"
#include <gtsam/navigation/AttitudeFactor.h>
#include <tbb/global_control.h>
#include <iostream>
#include <limits>
namespace n3mapping {
bool GraphOptimizer::loadGraph(
    const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes,
    const std::vector<EdgeInfo>& edges) {
    return loadGraph(nodes, edges, {});
}

bool GraphOptimizer::loadGraph(
    const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes,
    const std::vector<EdgeInfo>& edges,
    const std::vector<FloorAttitudeConstraint>& floor_constraints,
    const std::vector<SharedPlane>& planes, bool restore_estimates,
    const Eigen::Isometry3d* prior_pose) {
    GraphOptimizer temp(config_);

    if (nodes.empty()) {
        swapWith(temp);
        return true;
    }

    // 添加节点
    for (const auto& [id, pose] : nodes) {
        gtsam::Key key = gtsam::Symbol('x', id);
        gtsam::Pose3 gtsam_pose = eigenToGtsam(pose);
        if (temp.initial_values_.exists(key)) {
            LOG(ERROR) << "GraphOptimizer::loadGraph() duplicate node id: " << id << std::endl;
            return false;
        }
        temp.initial_values_.insert(key, gtsam_pose);
        temp.node_ids_.insert(id);
    }

    // 添加第一个节点的先验因子
    int64_t first_id = nodes.front().first;
    gtsam::Key first_key = gtsam::Symbol('x', first_id);
    gtsam::Pose3 first_pose = prior_pose ? eigenToGtsam(*prior_pose) :
        temp.initial_values_.at<gtsam::Pose3>(first_key);
    auto prior_noise = temp.createPriorNoiseModel();
    temp.graph_.add(gtsam::PriorFactor<gtsam::Pose3>(first_key, first_pose, prior_noise));

    int64_t first_session_node_id = std::numeric_limits<int64_t>::max();
    for (const auto& edge : edges) {
        if (edge.type == EdgeType::SESSION_ANCHOR) {
            first_session_node_id =
                std::min(first_session_node_id, edge.to_id);
        }
    }

    // 添加边
    for (const auto& edge : edges) {
        gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
        gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
        gtsam::Pose3 measurement = eigenToGtsam(edge.measurement);

        gtsam::noiseModel::Base::shared_ptr noise_model;
        const bool robust_session_odometry =
            edge.type == EdgeType::ODOMETRY &&
            first_session_node_id != std::numeric_limits<int64_t>::max() &&
            edge.from_id >= first_session_node_id &&
            edge.to_id >= first_session_node_id;
        if (isRobustGlobalEdge(edge.type) || robust_session_odometry) {
            noise_model = edge.constraint_mode == EdgeConstraintMode::XY_YAW
                ? temp.createXYYawLoopNoiseModel(edge.information, config_.use_robust_kernel)
                : temp.createRobustNoiseModel(edge.information, config_.use_robust_kernel);
            if (!noise_model) {
                noise_model = temp.createLoopNoiseModel();
            }
            if (edge.type == EdgeType::SESSION_ANCHOR) {
                temp.has_session_anchor_ = true;
            } else if (edge.type == EdgeType::LOOP) {
                temp.has_loop_closure_ = true;
            }
        } else {
            noise_model = createNoiseModel(edge.information);
            if (!noise_model) {
                noise_model = temp.createOdomNoiseModel();
            }
        }

        if (isRobustGlobalEdge(edge.type) && edge.constraint_mode == EdgeConstraintMode::XY_YAW) {
            temp.graph_.add(Pose3XYYawFactor(key_from, key_to, measurement, noise_model));
        } else {
            temp.graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
        }
        temp.edges_.push_back(edge);
    }

    // Load committed floor attitude constraints (validated by the
    // serializer; defensive checks keep a bad record from breaking the
    // load). The node must exist and the normal/sigma must be sane.
    for (const auto& fc : floor_constraints) {
        if (!fc.normal_body.allFinite() || fc.normal_body.norm() < 1e-6 ||
            !std::isfinite(fc.sigma_rad) || fc.sigma_rad <= 0.0) {
            continue;
        }
        gtsam::Key key = gtsam::Symbol('x', fc.node_id);
        if (!temp.initial_values_.exists(key)) {
            continue;
        }
        auto noise = gtsam::noiseModel::Isotropic::Sigma(2, fc.sigma_rad);
        temp.graph_.add(gtsam::Pose3AttitudeFactor(
            key, gtsam::Unit3(0.0, 0.0, 1.0), noise,
            gtsam::Unit3(fc.normal_body.normalized())));
        temp.committed_floor_attitude_constraints_.push_back(fc);
        ++temp.floor_attitude_factor_count_;
    }

    // 执行优化
    if (!restore_estimates && planes.empty() &&
        !temp.graph_.empty() && !temp.initial_values_.empty()) {
        gtsam::Values optimized_estimate;
        // The installed GTSAM uses TBB for Bayes-tree back-substitution. On
        // large cross-linked saved graphs its parallel traversal can race a
        // child clique against publication of its parent's result and throw
        // std::out_of_range ("map::at"). Keep only this one-shot bulk reload
        // solve serial; online incremental optimization and point-cloud work
        // retain their configured parallelism.
        tbb::global_control serial_reload_solve(
            tbb::global_control::max_allowed_parallelism, 1);
        if (!temp.optimizeCandidate(temp.graph_, temp.initial_values_, &optimized_estimate)) {
            LOG(ERROR) << "GraphOptimizer::loadGraph() failed; existing graph left unchanged" << std::endl;
            return false;
        }
        temp.current_estimate_ = optimized_estimate;
        temp.last_good_estimate_ = optimized_estimate;
    }

    if (restore_estimates || !planes.empty()) {
        temp.current_estimate_ = temp.initial_values_;
        temp.last_good_estimate_ = temp.initial_values_;
        GeometryOptimizationStats stats;
        if (!temp.replaceSharedPlanes(planes, false, stats)) return false;
    } else {
        temp.rebuildISAM2FromCommitted();
    }
    temp.clearPending();
    temp.needs_optimization_ = false;
    swapWith(temp);
    return true;
}

} // namespace n3mapping
