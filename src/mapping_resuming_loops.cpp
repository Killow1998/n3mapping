#include "n3mapping/mapping_resuming.h"
#include "n3mapping/world_localizing.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/rotation_utils.h"
#include <glog/logging.h>
#include <algorithm>
#include "n3mapping/loop_verification_pipeline.h"

namespace n3mapping {

int MappingResuming::detectCrossLoops(int64_t new_keyframe_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != MappingResumingState::EXTENDING ||
        isFromOriginalMap(new_keyframe_id) ||
        !optimizer_.hasNode(new_keyframe_id)) {
        return 0;
    }

    auto new_kf = keyframe_manager_.getKeyframe(new_keyframe_id);
    if (!new_kf || !new_kf->cloud) return 0;

    // The first new keyframe is already tied to the loaded map by the explicit
    // SESSION_ANCHOR. Re-registering that same frame immediately is redundant
    // and, before this guard, allowed a raw ICP Hessian to overpower the anchor.
    if (new_keyframe_id == original_max_keyframe_id_ + 1) {
        return 0;
    }

    std::map<int64_t, Keyframe::Ptr> keyframe_map;
    for (const auto& keyframe : keyframe_manager_.getAllKeyframes()) {
        if (keyframe) {
            keyframe_map[keyframe->id] = keyframe;
        }
    }
    const auto accept_loaded_map = [&](int64_t match_id) {
        const auto it = keyframe_map.find(match_id);
        return it != keyframe_map.end() && it->second &&
               it->second->is_from_loaded_map;
    };
    auto candidates = loop_detector_.detectLoopCandidates(
        new_keyframe_id, keyframe_map, accept_loaded_map);
    if (config_.loop_spatial_candidates_enable) {
        auto spatial_candidates = loop_detector_.detectSpatialCandidates(
            new_keyframe_id, keyframe_map, accept_loaded_map);
        for (const auto& spatial : spatial_candidates) {
            auto duplicate = std::find_if(
                candidates.begin(), candidates.end(),
                [&](const LoopCandidate& existing) {
                    return existing.query_id == spatial.query_id &&
                           existing.match_id == spatial.match_id;
                });
            if (duplicate == candidates.end()) {
                candidates.push_back(spatial);
            } else {
                duplicate->source_flags |= spatial.source_flags;
                duplicate->spatial_score =
                    std::max(duplicate->spatial_score, spatial.spatial_score);
            }
        }
    }

    std::vector<VerifiedLoop> verified_loops;
    verified_loops.reserve(candidates.size());
    LoopVerificationPipeline verification_pipeline(
        config_, keyframe_manager_, matcher_, optimizer_,
        loop_closure_manager_);
    LoopVerificationContext verification_context;
    verification_context.cross_session = true;
    verification_context.pose_visibility_evaluator =
        [this, query_cloud = new_kf->cloud](
            const Eigen::Isometry3d& T_map_query) {
            return world_localizing_.evaluateLoadedMapPoseVisibility(
                query_cloud, T_map_query);
        };

    for (const auto& candidate : candidates) {
        if (!isFromOriginalMap(candidate.match_id) ||
            !accept_loaded_map(candidate.match_id)) continue;

        auto verification = verification_pipeline.evaluate(
            candidate, verification_context);
        if (!verification.loop.verified) {
            VLOG(1) << "[MappingResuming] Reject cross-session candidate query="
                    << candidate.query_id << " match=" << candidate.match_id
                    << " stage=" << verification.reject_stage
                    << " reason=" << verification.reject_reason
                    << " fitness="
                    << verification.verification.match_result.fitness_score
                    << " inlier="
                    << verification.verification.match_result.inlier_ratio
                    << " descriptor_seeded="
                    << verification.descriptor_seeded
                    << " yaw_seed=" << verification.selected_seed_yaw_rad
                    << " hypotheses="
                    << verification.registration_hypothesis_count
                    << " visibility="
                    << verification.pose_visibility.consistency_ratio
                    << " visibility_logodds="
                    << verification.pose_visibility.evidence_log_odds
                    << " visibility_conflict="
                    << verification.pose_visibility.foreground_conflict_ratio;
            continue;
        }
        VLOG(1) << "[MappingResuming] Verified cross-session candidate query="
                << candidate.query_id << " match=" << candidate.match_id
                << " fitness="
                << verification.verification.match_result.fitness_score
                << " inlier="
                << verification.verification.match_result.inlier_ratio
                << " descriptor_seeded="
                << verification.descriptor_seeded
                << " yaw_seed=" << verification.selected_seed_yaw_rad
                << " hypotheses="
                << verification.registration_hypothesis_count
                << " visibility="
                << verification.pose_visibility.consistency_ratio
                << " visibility_logodds="
                << verification.pose_visibility.evidence_log_odds
                << " visibility_conflict="
                << verification.pose_visibility.foreground_conflict_ratio;
        verified_loops.push_back(std::move(verification.loop));
    }

    auto valid_loops = loop_closure_manager_.filterValidLoops(verified_loops);
    std::vector<LoopConstraintPipelineResult> accepted_constraints;
    accepted_constraints.reserve(valid_loops.size());
    const LoopConstraintContext constraint_context{true};
    // A good single-frame score is not enough to choose the cross-session
    // correspondence. Gate every bounded candidate with independent neighbor
    // evidence first; otherwise one attractive alias can be selected, fail
    // consensus, and suppress a valid candidate from the same query.
    for (auto& constraint : verification_pipeline.evaluateConstraints(
             valid_loops, LoopEdgeDirection::MatchToQuery, constraint_context)) {
        const auto& loop = constraint.loop;
        VLOG(1) << "[MappingResuming] Cross-session consensus query="
                << loop.query_id << " match=" << loop.match_id
                << " decision="
                << loopConsensusDecisionName(constraint.consensus.decision)
                << " valid=" << constraint.consensus.valid_pair_count
                << " left=" << constraint.consensus.left_support_count
                << " right=" << constraint.consensus.right_support_count
                << " contradictions="
                << constraint.consensus.contradiction_count
                << " median_t="
                << constraint.consensus.median_translation_delta
                << " median_r="
                << constraint.consensus.median_rotation_delta
                << " estimator="
                << constraint.consensus.estimator_recommendation
                << " estimator_pairs="
                << constraint.consensus.estimator_pair_count;
        if (!constraint.accepted || !constraint.has_edge) {
            VLOG(1) << "[MappingResuming] Defer cross-session constraint query="
                    << loop.query_id << " match=" << loop.match_id
                    << " stage=" << constraint.reject_stage
                    << " reason=" << constraint.reject_reason
                    << " consensus="
                    << loopConsensusDecisionName(constraint.consensus.decision);
            continue;
        }
        accepted_constraints.push_back(std::move(constraint));
    }

    std::vector<VerifiedLoop> accepted_loops;
    accepted_loops.reserve(accepted_constraints.size());
    for (const auto& constraint : accepted_constraints) {
        accepted_loops.push_back(constraint.loop);
    }
    const auto best_loops =
        loop_closure_manager_.selectBestPerQuery(accepted_loops);
    std::vector<EdgeInfo> edges;
    edges.reserve(best_loops.size());
    for (const auto& best : best_loops) {
        const auto selected = std::find_if(
            accepted_constraints.begin(), accepted_constraints.end(),
            [&](const LoopConstraintPipelineResult& candidate) {
                return candidate.loop.query_id == best.query_id &&
                       candidate.loop.match_id == best.match_id;
            });
        if (selected != accepted_constraints.end()) {
            edges.push_back(selected->edge);
        }
    }
    if (!edges.empty()) {
        // A correct session-merge loop can be tens of metres from the raw
        // odometry chain. With a redescending kernel it would enter as an
        // outlier and barely move the graph. Seed the new-session component
        // at the accepted measurement, preserving every raw odometry edge,
        // then rebuild transactionally so the robust loop starts in-basin.
        if (edges.size() != 1) {
            LOG(ERROR) << "[MappingResuming] Expected one selected cross-session "
                          "edge, got " << edges.size();
            return 0;
        }
        const auto poses_before_rebase = optimizer_.getOptimizedPoses();
        const auto from_pose = poses_before_rebase.find(edges.front().from_id);
        const auto to_pose = poses_before_rebase.find(edges.front().to_id);
        if (from_pose == poses_before_rebase.end() ||
            to_pose == poses_before_rebase.end()) {
            return 0;
        }
        const Eigen::Isometry3d rebase_correction =
            from_pose->second * edges.front().measurement *
            to_pose->second.inverse();
        const int64_t first_unconstrained_node =
            std::max(original_max_keyframe_id_ + 1,
                     last_trusted_constraint_keyframe_id_ + 1);
        if (!optimizer_.rebaseSessionAndAddLoopEdge(
                edges.front(), first_unconstrained_node)) {
            LOG(WARNING) << "[MappingResuming] Cross-session graph rebase failed "
                         << "query=" << new_keyframe_id;
            return 0;
        }
        LOG(INFO) << "[MappingResuming] Rebased new session before loop commit "
                  << "query=" << new_keyframe_id
                  << " correction_t="
                  << rebase_correction.translation().norm()
                  << " correction_r="
                  << Eigen::AngleAxisd(rebase_correction.rotation()).angle();
        cross_loop_count_ += edges.size();
        last_trusted_constraint_keyframe_id_ = new_keyframe_id;
        LOG(INFO) << "[MappingResuming] Committed cross-session loop query="
                  << new_keyframe_id << " edge_count=" << edges.size();

        auto optimized_poses = optimizer_.getOptimizedPoses();
        keyframe_manager_.updateOptimizedPoses(optimized_poses);
        refreshSubmapPosesNoLock("cross_session_loop");

        // Map-to-odom is a live correction, not a once-per-session constant.
        // Once a trusted old-map constraint moves the current query in the
        // graph, carry that correction into every non-keyframe output and the
        // initial value of the next keyframe while retaining raw odometry on
        // the ODOMETRY edge itself.
        const auto optimized_it = optimized_poses.find(new_keyframe_id);
        if (optimized_it != optimized_poses.end() &&
            previous_new_keyframe_id_ == new_keyframe_id &&
            previous_new_odom_pose_.matrix().allFinite()) {
            const Eigen::Isometry3d corrected_map_odom =
                optimized_it->second * previous_new_odom_pose_.inverse();
            if (corrected_map_odom.matrix().allFinite()) {
                world_localizing_.setMapToOdomTransform(corrected_map_odom);
                const Eigen::Vector3d corrected_rpy_deg =
                    (canonicalRollPitchYaw(corrected_map_odom) * (180.0 / M_PI));
                LOG(INFO) << "[MappingResuming] Updated T_map_odom from "
                             "cross-session graph query="
                          << new_keyframe_id << " t="
                          << corrected_map_odom.translation().transpose()
                          << " rpy_deg=" << corrected_rpy_deg.transpose();
            }
        }
    }

    return static_cast<int>(edges.size());
}

}  // namespace n3mapping
