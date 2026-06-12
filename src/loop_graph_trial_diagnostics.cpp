#include "n3mapping/loop_graph_trial_diagnostics.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>

namespace n3mapping {
namespace {

double rotationAngle(const Eigen::Isometry3d& transform)
{
    return std::abs(Eigen::AngleAxisd(transform.rotation()).angle());
}

Eigen::Vector3d rollPitchYawAbs(const Eigen::Isometry3d& transform)
{
    return transform.rotation().eulerAngles(0, 1, 2).cwiseAbs();
}

double safeFinite(double value, double fallback = 0.0)
{
    return std::isfinite(value) ? value : fallback;
}

double meanResidualScore(const std::vector<EdgeInfo>& edges,
                         const std::map<int64_t, Eigen::Isometry3d>& poses,
                         EdgeType type)
{
    double sum = 0.0;
    std::size_t count = 0;
    for (const auto& edge : edges) {
        if (edge.type != type) {
            continue;
        }
        auto from_it = poses.find(edge.from_id);
        auto to_it = poses.find(edge.to_id);
        if (from_it == poses.end() || to_it == poses.end()) {
            continue;
        }
        const Eigen::Isometry3d predicted = from_it->second.inverse() * to_it->second;
        const Eigen::Isometry3d residual = edge.measurement.inverse() * predicted;
        sum += residual.translation().norm() + rotationAngle(residual);
        ++count;
    }
    if (count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / static_cast<double>(count);
}

void fillCandidateResiduals(const std::vector<EdgeInfo>& candidate_edges,
                            const std::map<int64_t, Eigen::Isometry3d>& poses,
                            LoopGraphTrialDiagnostics* diagnostics)
{
    if (!diagnostics || candidate_edges.empty()) {
        return;
    }
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d rpy = Eigen::Vector3d::Zero();
    double translation_norm = 0.0;
    double rotation_norm = 0.0;
    std::size_t count = 0;

    for (const auto& edge : candidate_edges) {
        auto from_it = poses.find(edge.from_id);
        auto to_it = poses.find(edge.to_id);
        if (from_it == poses.end() || to_it == poses.end()) {
            continue;
        }
        const Eigen::Isometry3d predicted = from_it->second.inverse() * to_it->second;
        const Eigen::Isometry3d residual = edge.measurement.inverse() * predicted;
        translation += residual.translation().cwiseAbs();
        rpy += rollPitchYawAbs(residual);
        translation_norm += residual.translation().norm();
        rotation_norm += rotationAngle(residual);
        ++count;
    }

    if (count == 0) {
        return;
    }

    const double inv_count = 1.0 / static_cast<double>(count);
    translation *= inv_count;
    rpy *= inv_count;
    diagnostics->residual_x_after = translation.x();
    diagnostics->residual_y_after = translation.y();
    diagnostics->residual_z_after = translation.z();
    diagnostics->residual_roll_after = rpy.x();
    diagnostics->residual_pitch_after = rpy.y();
    diagnostics->residual_yaw_after = rpy.z();
    diagnostics->residual_translation_norm_after = translation_norm * inv_count;
    diagnostics->residual_rotation_norm_after = rotation_norm * inv_count;
}

void fillPoseUpdateStats(const std::map<int64_t, Eigen::Isometry3d>& before,
                         const std::map<int64_t, Eigen::Isometry3d>& after,
                         LoopGraphTrialDiagnostics* diagnostics)
{
    if (!diagnostics) {
        return;
    }
    double translation_sum = 0.0;
    double rotation_sum = 0.0;
    double max_translation = 0.0;
    double max_rotation = 0.0;
    std::size_t count = 0;

    for (const auto& [id, before_pose] : before) {
        auto after_it = after.find(id);
        if (after_it == after.end()) {
            continue;
        }
        const Eigen::Isometry3d delta = before_pose.inverse() * after_it->second;
        const double translation = delta.translation().norm();
        const double rotation = rotationAngle(delta);
        translation_sum += translation;
        rotation_sum += rotation;
        max_translation = std::max(max_translation, translation);
        max_rotation = std::max(max_rotation, rotation);
        ++count;
    }

    if (count == 0) {
        return;
    }
    const double inv_count = 1.0 / static_cast<double>(count);
    diagnostics->mean_pose_update_translation = translation_sum * inv_count;
    diagnostics->max_pose_update_translation = max_translation;
    diagnostics->mean_pose_update_rotation = rotation_sum * inv_count;
    diagnostics->max_pose_update_rotation = max_rotation;
}

double consistencyScore(const LoopGraphTrialDiagnostics& diagnostics)
{
    if (!diagnostics.success) {
        return 0.0;
    }
    const double penalty =
        safeFinite(diagnostics.residual_translation_norm_after) +
        safeFinite(diagnostics.residual_rotation_norm_after) +
        safeFinite(diagnostics.max_pose_update_translation) +
        safeFinite(diagnostics.max_pose_update_rotation) +
        std::max(0.0, safeFinite(diagnostics.existing_loop_residual_delta)) +
        std::max(0.0, safeFinite(diagnostics.odom_residual_delta));
    return 1.0 / (1.0 + penalty);
}

}  // namespace

LoopGraphTrialDiagnostics computeLoopGraphTrialDiagnostics(
    const Config& config,
    const std::map<int64_t, Eigen::Isometry3d>& poses_before,
    const std::vector<EdgeInfo>& committed_edges,
    const std::vector<EdgeInfo>& candidate_edges)
{
    LoopGraphTrialDiagnostics diagnostics;
    diagnostics.recommendation = "trial_not_run";
    if (poses_before.empty() || candidate_edges.empty()) {
        diagnostics.recommendation = "trial_unavailable";
        return diagnostics;
    }

    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
    nodes.reserve(poses_before.size());
    for (const auto& [id, pose] : poses_before) {
        nodes.emplace_back(id, pose);
    }

    try {
        GraphOptimizer trial_optimizer(config);
        if (!trial_optimizer.loadGraph(nodes, committed_edges)) {
            diagnostics.recommendation = "trial_load_failed";
            return diagnostics;
        }
        for (const auto& edge : candidate_edges) {
            trial_optimizer.addLoopEdge(edge);
        }
        if (!trial_optimizer.incrementalOptimize()) {
            diagnostics.recommendation = "trial_optimize_failed";
            return diagnostics;
        }
        const auto poses_after = trial_optimizer.getOptimizedPoses();
        diagnostics.success = true;
        fillCandidateResiduals(candidate_edges, poses_after, &diagnostics);
        fillPoseUpdateStats(poses_before, poses_after, &diagnostics);

        const double existing_loop_before = meanResidualScore(committed_edges, poses_before, EdgeType::LOOP);
        const double existing_loop_after = meanResidualScore(committed_edges, poses_after, EdgeType::LOOP);
        if (std::isfinite(existing_loop_before) && std::isfinite(existing_loop_after)) {
            diagnostics.existing_loop_residual_delta = existing_loop_after - existing_loop_before;
        }
        const double odom_before = meanResidualScore(committed_edges, poses_before, EdgeType::ODOMETRY);
        const double odom_after = meanResidualScore(committed_edges, poses_after, EdgeType::ODOMETRY);
        if (std::isfinite(odom_before) && std::isfinite(odom_after)) {
            diagnostics.odom_residual_delta = odom_after - odom_before;
        }
        diagnostics.consistency_score = consistencyScore(diagnostics);
        diagnostics.recommendation = "trial_success_score_only";
    } catch (const std::exception&) {
        diagnostics.success = false;
        diagnostics.recommendation = "trial_exception";
    }
    return diagnostics;
}

}  // namespace n3mapping
