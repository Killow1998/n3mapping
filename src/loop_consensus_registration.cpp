// Independent neighbor measurements, grouped by their unchanged target geometry.
#include "n3mapping/loop_consensus_verifier.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace n3mapping {
namespace {
double rotationAngle(const Eigen::Isometry3d& transform) {
    return Eigen::AngleAxisd(transform.rotation()).angle();
}

Keyframe::Ptr findLoadedMapCorrespondence(
    const std::vector<Keyframe::Ptr>& keyframes, int64_t central_match_id,
    int local_id_radius, const Eigen::Isometry3d& expected_world_pose)
{
    Keyframe::Ptr best;
    double best_score = std::numeric_limits<double>::infinity();
    for (const auto& keyframe : keyframes) {
        if (!keyframe || !keyframe->is_from_loaded_map ||
            std::abs(keyframe->id - central_match_id) > local_id_radius) {
            continue;
        }
        const Eigen::Isometry3d residual =
            expected_world_pose.inverse() * keyframe->pose_optimized;
        const double score = residual.translation().norm() +
                             rotationAngle(residual);
        if (score < best_score) {
            best_score = score;
            best = keyframe;
        }
    }
    return best;
}

struct NeighborTask {
    std::size_t loop_index;
    std::size_t pair_index;
    Keyframe::Ptr query;
    Keyframe::Ptr match;
    Keyframe::Ptr query_neighbor;
    Keyframe::Ptr match_neighbor;
    Eigen::Isometry3d predicted;
};
}  // namespace

std::vector<LoopConsensusResult> LoopConsensusVerifier::evaluate(
    const KeyframeManager& keyframes, PointCloudMatcher& matcher,
    const std::vector<VerifiedLoop>& central_loops,
    int half_window, bool cross_session) const {
    std::vector<std::vector<LoopConsensusPairEvidence>> pairs(central_loops.size());
    std::map<int64_t, std::vector<NeighborTask>> tasks_by_target;
    const auto all_keyframes = keyframes.getAllKeyframes();
    const int local_id_radius = std::max(12, 6 * half_window);
    for (std::size_t loop_index = 0; loop_index < central_loops.size(); ++loop_index) {
        const auto& central_loop = central_loops[loop_index];
        if (half_window < 1 || central_loop.query_id < 0 || central_loop.match_id < 0)
            continue;
        const auto query = keyframes.getKeyframe(central_loop.query_id);
        const auto match = keyframes.getKeyframe(central_loop.match_id);
        if (!query || !match) continue;
        auto& loop_pairs = pairs[loop_index];
        for (int offset = -half_window; offset <= half_window; ++offset) {
            if (offset == 0) continue;
            loop_pairs.emplace_back();
            auto& evidence = loop_pairs.back();
            evidence.offset = offset;
            evidence.query_neighbor_id = central_loop.query_id + offset;
            const auto query_neighbor = keyframes.getKeyframe(evidence.query_neighbor_id);
            if (!query_neighbor) {
                evidence.reject_reason = "missing_or_recent_neighbor";
                continue;
            }

            Keyframe::Ptr match_neighbor;
            Eigen::Isometry3d predicted = Eigen::Isometry3d::Identity();
            if (cross_session) {
                // Separate sessions need pose-based correspondence: their
                // keyframe IDs do not describe equal sampling intervals.
                if (query_neighbor->is_from_loaded_map) {
                    evidence.reject_reason = "session_boundary_neighbor";
                    continue;
                }
                const Eigen::Isometry3d T_query_query_neighbor =
                    query->pose_optimized.inverse() * query_neighbor->pose_optimized;
                const Eigen::Isometry3d expected_world_pose =
                    match->pose_optimized * central_loop.T_measured_match_query *
                    T_query_query_neighbor;
                match_neighbor = findLoadedMapCorrespondence(
                    all_keyframes, central_loop.match_id, local_id_radius, expected_world_pose);
                if (!match_neighbor) {
                    evidence.reject_reason = "missing_loaded_map_correspondence";
                    continue;
                }
                evidence.match_neighbor_id = match_neighbor->id;
                predicted = match_neighbor->pose_optimized.inverse() * expected_world_pose;
            } else {
                evidence.match_neighbor_id = central_loop.match_id + offset;
                match_neighbor = keyframes.getKeyframe(evidence.match_neighbor_id);
                if (!match_neighbor ||
                    std::abs(evidence.query_neighbor_id - evidence.match_neighbor_id) <
                        config_.sc_num_exclude_recent) {
                    evidence.reject_reason = "missing_or_recent_neighbor";
                    continue;
                }
                predicted = predictNeighborTransform(
                    query->pose_optimized, match->pose_optimized,
                    query_neighbor->pose_optimized, match_neighbor->pose_optimized,
                    central_loop.T_measured_match_query);
            }
            if (cross_session && !match_neighbor->is_from_loaded_map) {
                evidence.reject_reason = "session_boundary_neighbor";
                continue;
            }
            tasks_by_target[match_neighbor->id].push_back({
                loop_index, loop_pairs.size() - 1, query, match,
                query_neighbor, match_neighbor, predicted});
        }
    }

    // Only one prepared target is resident. Solves and evidence remain per
    // central candidate; changing execution order cannot pool their votes.
    for (const auto& [target_id, tasks] : tasks_by_target) {
        const auto target = keyframes.buildSubmapInRootFrame(
            target_id, config_.gicp_submap_size, target_id);
        PointCloudMatcher::PreparedTarget prepared_target;
        bool prepared = false;
        for (const auto& task : tasks) {
            auto& evidence = pairs[task.loop_index][task.pair_index];
            const auto source = keyframes.buildSubmapInRootFrame(
                evidence.query_neighbor_id, 0, evidence.query_neighbor_id);
            if (!source || source->size() < 10 || !target || target->size() < 10) {
                evidence.reject_reason = "empty_neighbor_submap";
                continue;
            }
            if (!prepared) {
                prepared_target = matcher.prepareTargetCloud(target);
                prepared = true;
            }
            const auto prepared_source = matcher.prepareSourceCloud(source);
            const auto match_result = matcher.alignPrepared(
                prepared_target, prepared_source, task.predicted);
            evidence.valid = match_result.num_inliers > 0;
            evidence.converged = match_result.converged;
            evidence.fitness_score = match_result.fitness_score;
            evidence.inlier_ratio = match_result.inlier_ratio;
            // An unchanged iterate can satisfy the solver's convergence test
            // with zero correspondences. That is no neighborhood observation,
            // so it must neither support nor contradict the central loop.
            if (!evidence.valid) {
                evidence.reject_reason = "no_neighbor_correspondences";
                continue;
            }
            if (match_result.converged) {
                const Eigen::Isometry3d delta =
                    task.predicted.inverse() * match_result.T_target_source;
                evidence.delta_translation_norm = delta.translation().norm();
                evidence.delta_rotation_norm = rotationAngle(delta);
                const Eigen::Isometry3d T_match_neighbor_match =
                    task.match_neighbor->pose_optimized.inverse() * task.match->pose_optimized;
                const Eigen::Isometry3d T_query_query_neighbor =
                    task.query->pose_optimized.inverse() * task.query_neighbor->pose_optimized;
                evidence.estimated_match_query = T_match_neighbor_match.inverse() *
                    match_result.T_target_source * T_query_query_neighbor.inverse();
                evidence.has_estimated_measurement = true;
            } else {
                evidence.reject_reason = "icp_not_converged";
            }
        }
    }

    std::vector<LoopConsensusResult> results;
    results.reserve(central_loops.size());
    for (std::size_t i = 0; i < central_loops.size(); ++i) {
        results.push_back(summarizePairs(
            config_, pairs[i], central_loops[i].T_measured_match_query));
    }
    return results;
}

}  // namespace n3mapping
