#include "n3mapping/graph_optimizer.h"
#include "graph_pose_factors.h"
#include <gtsam/navigation/AttitudeFactor.h>
#include <algorithm>
#include <iostream>
namespace n3mapping {
void GraphOptimizer::addPriorFactor(int64_t id, const Eigen::Isometry3d& pose) {
    gtsam::Key key = gtsam::Symbol('x', id);
    gtsam::Pose3 gtsam_pose = eigenToGtsam(pose);

    // 创建先验噪声模型
    auto noise_model = createPriorNoiseModel();

    // 添加到待提交因子；只有优化成功后才进入 committed graph。
    new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(key, gtsam_pose, noise_model));

    // 添加待提交初始值
    if (!hasAnyNode(id)) {
        new_values_.insert(key, gtsam_pose);
        pending_node_ids_.insert(id);
    }

    needs_optimization_ = true;
}

void GraphOptimizer::addOdometryEdge(const EdgeInfo& edge) {
    gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
    gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
    gtsam::Pose3 measurement = eigenToGtsam(edge.measurement);

    // 创建噪声模型
    auto noise_model = createNoiseModel(edge.information);
    if (!noise_model) {
        noise_model = createOdomNoiseModel();
    }

    // 添加到待提交因子；只有优化成功后才进入 committed graph。
    new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));

    // 如果目标节点不存在，尝试从 from 节点推导待提交初始值。
    (void)ensureTargetInitialValue(
        edge.from_id, edge.to_id, edge.measurement);

    // 存储待提交边信息
    pending_edges_.push_back(edge);
    needs_optimization_ = true;
}

bool GraphOptimizer::addSessionOdometryEdge(
    const EdgeInfo& edge,
    const Eigen::Isometry3d& target_initial_pose) {
    if (edge.from_id < 0 || edge.to_id < 0 || edge.from_id == edge.to_id ||
        !edge.measurement.matrix().allFinite() ||
        !edge.information.allFinite() ||
        !target_initial_pose.matrix().allFinite() ||
        !hasAnyNode(edge.from_id) || hasAnyNode(edge.to_id)) {
        return false;
    }

    const gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
    const gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
    auto noise_model = createRobustNoiseModel(
        edge.information, config_.use_robust_kernel);
    new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
        key_from, key_to, eigenToGtsam(edge.measurement), noise_model));
    new_values_.insert(key_to, eigenToGtsam(target_initial_pose));
    pending_node_ids_.insert(edge.to_id);

    EdgeInfo odometry_edge = edge;
    odometry_edge.type = EdgeType::ODOMETRY;
    pending_edges_.push_back(odometry_edge);
    needs_optimization_ = true;
    return true;
}


void GraphOptimizer::addLoopEdge(const EdgeInfo& edge) {
    gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
    gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
    gtsam::Pose3 measurement = eigenToGtsam(edge.measurement);

    // 创建带鲁棒核函数的噪声模型
    auto noise_model = edge.constraint_mode == EdgeConstraintMode::XY_YAW
        ? createXYYawLoopNoiseModel(edge.information, config_.use_robust_kernel)
        : createRobustNoiseModel(edge.information, config_.use_robust_kernel);

    // 添加到待提交因子；只有优化成功后才进入 committed graph。
    if (edge.constraint_mode == EdgeConstraintMode::XY_YAW) {
        new_factors_.add(Pose3XYYawFactor(key_from, key_to, measurement, noise_model));
    } else {
        new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
    }

    // 存储待提交边信息
    EdgeInfo loop_edge = edge;
    loop_edge.type = EdgeType::LOOP;
    pending_edges_.push_back(loop_edge);

    pending_has_loop_closure_ = true;
    needs_optimization_ = true;
}

bool GraphOptimizer::rebaseSessionAndAddLoopEdge(
    const EdgeInfo& edge, int64_t first_session_node_id) {
    if (needs_optimization_ || first_session_node_id < 0 ||
        edge.from_id < 0 || edge.to_id < first_session_node_id ||
        edge.from_id >= first_session_node_id ||
        edge.constraint_mode != EdgeConstraintMode::FULL_6DOF ||
        !edge.measurement.matrix().allFinite() ||
        !edge.information.allFinite()) {
        return false;
    }

    const auto poses = getOptimizedPoses();
    const auto from = poses.find(edge.from_id);
    const auto to = poses.find(edge.to_id);
    if (from == poses.end() || to == poses.end()) {
        return false;
    }
    const Eigen::Isometry3d desired_to =
        from->second * edge.measurement;
    const Eigen::Isometry3d correction =
        desired_to * to->second.inverse();
    if (!correction.matrix().allFinite()) {
        return false;
    }

    std::vector<std::pair<int64_t, Eigen::Isometry3d>> rebased_nodes;
    rebased_nodes.reserve(poses.size());
    for (const auto& [id, pose] : poses) {
        rebased_nodes.emplace_back(
            id, id >= first_session_node_id ? correction * pose : pose);
    }
    auto rebuilt_edges = edges_;
    EdgeInfo loop_edge = edge;
    loop_edge.type = EdgeType::LOOP;
    rebuilt_edges.push_back(loop_edge);
    if (shared_planes_.empty())
        return loadGraph(rebased_nodes, rebuilt_edges, committed_floor_attitude_constraints_);
    auto planes = sharedPlanes();
    // A large cross-session rigid reset invalidates that session's associations.
    // Keep old-map observations; the next geometry job reassociates the new ones.
    for (auto& plane : planes)
        plane.observations.erase(std::remove_if(plane.observations.begin(), plane.observations.end(),
            [&](const auto& o) { return o.node_id >= first_session_node_id; }), plane.observations.end());
    planes.erase(std::remove_if(planes.begin(), planes.end(),
        [](const auto& p) { return p.observations.size() < 3; }), planes.end());
    GraphOptimizer candidate(config_);
    const auto prior = gaugePrior();
    GeometryOptimizationStats stats;
    if (!candidate.loadGraph(rebased_nodes, rebuilt_edges, committed_floor_attitude_constraints_,
                             planes, true, &prior) ||
        !candidate.replaceSharedPlanes(planes, true, stats)) return false;
    swapWith(candidate);
    return true;
}

bool GraphOptimizer::addSessionAnchorEdge(const EdgeInfo& edge) {
    if (edge.from_id < 0 || edge.to_id < 0 ||
        !edge.measurement.matrix().allFinite() ||
        !edge.information.allFinite()) {
        return false;
    }

    gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
    gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
    gtsam::Pose3 measurement = eigenToGtsam(edge.measurement);

    // Same robust-noise path as loop closures: a cross-session anchor is a
    // global constraint, not an odometry chain link.
    auto noise_model = edge.constraint_mode == EdgeConstraintMode::XY_YAW
        ? createXYYawLoopNoiseModel(edge.information, config_.use_robust_kernel)
        : createRobustNoiseModel(edge.information, config_.use_robust_kernel);

    // Unlike a loop, the first cross-session anchor may introduce a new graph
    // node. Reject a missing source before staging a factor or edge.
    if (!ensureTargetInitialValue(
            edge.from_id, edge.to_id, edge.measurement)) {
        return false;
    }

    if (edge.constraint_mode == EdgeConstraintMode::XY_YAW) {
        new_factors_.add(Pose3XYYawFactor(key_from, key_to, measurement, noise_model));
    } else {
        new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
    }

    EdgeInfo anchor_edge = edge;
    anchor_edge.type = EdgeType::SESSION_ANCHOR;
    pending_edges_.push_back(anchor_edge);

    pending_has_session_anchor_ = true;
    needs_optimization_ = true;
    return true;
}

bool GraphOptimizer::isRobustGlobalEdge(EdgeType type) {
    return type == EdgeType::LOOP || type == EdgeType::SESSION_ANCHOR;
}

bool GraphOptimizer::hasGlobalConstraint() const {
    return hasLoopClosure() || has_session_anchor_ || pending_has_session_anchor_;
}

} // namespace n3mapping
