#include "n3mapping/rotation_utils.h"
#include "n3mapping/loop_verification_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/loop_referee.h"
#include "n3mapping/loop_segment_consistency.h"
#include "constraint_refinement.h"

namespace n3mapping {
namespace {

double unitScoreBelow(double value, double limit) {
    if (!std::isfinite(value) || limit <= 0.0) {
        return 0.0;
    }
    return std::clamp(1.0 - value / limit, 0.0, 1.0);
}

LoopFeatures makeSegmentAwareLoopFeatures(const Config& config,
                                           const LoopCandidate& candidate,
                                           const VerifiedLoop& loop,
                                           double icp_translation_norm) {
    LoopFeatures features;
    features.descriptor_score = candidate.descriptor_score;
    features.spatial_score = candidate.spatial_score;
    features.geometric_overlap = loop.heightmap_vertical_consistency_score;
    features.temporal_gap = std::clamp(
        static_cast<double>(candidate.query_id - candidate.match_id) /
            std::max(1.0, static_cast<double>(config.sc_num_exclude_recent)),
        0.0, 1.0);
    const double fitness_score =
        unitScoreBelow(loop.fitness_score, config.loop_fitness_threshold);
    const double inlier_score =
        config.loop_min_inlier_ratio > 0.0
            ? std::clamp(loop.inlier_ratio / config.loop_min_inlier_ratio,
                         0.0, 1.0)
            : std::clamp(loop.inlier_ratio, 0.0, 1.0);
    const double motion_score = unitScoreBelow(
        icp_translation_norm, config.loop_max_icp_translation);
    features.local_map_consistency =
        (fitness_score + inlier_score + motion_score) / 3.0;
    features.segment_consistency = loop.segment_consensus_ratio;
    features.segment_support =
        loop.segment_pair_count > 0
            ? static_cast<double>(loop.segment_valid_pair_count) /
                  static_cast<double>(loop.segment_pair_count)
            : 0.0;
    const bool from_descriptor = candidate.fromRHPD() || candidate.fromSC();
    features.descriptor_supported = from_descriptor;
    features.spatial_only = candidate.fromSpatial() && !from_descriptor;
    features.predicted_translation_norm =
        loop.T_pred_match_query.translation().norm();
    features.icp_correction_yaw_abs =
        std::abs(canonicalRollPitchYaw(loop.T_icp_correction_match).z());
    features.segment_translation_median = loop.segment_translation_median;
    return features;
}

Keyframe::PointCloudT::Ptr strideLimitCloud(
    const Keyframe::PointCloudT::Ptr& cloud, std::size_t max_points) {
    if (!cloud || cloud->size() <= max_points || max_points == 0) {
        return cloud;
    }

    auto limited = pcl::make_shared<Keyframe::PointCloudT>();
    limited->header = cloud->header;
    limited->reserve(max_points);
    const std::size_t stride = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(
               static_cast<double>(cloud->size()) /
               static_cast<double>(max_points))));
    for (std::size_t i = 0;
         i < cloud->size() && limited->size() < max_points; i += stride) {
        const auto& point = cloud->points[i];
        if (isFinitePoint(point)) {
            limited->push_back(point);
        }
    }
    limited->width = static_cast<std::uint32_t>(limited->size());
    limited->height = 1;
    limited->is_dense = true;
    return limited;
}

Keyframe::PointCloudT::Ptr prepareLoopIcpCloud(
    const Keyframe::PointCloudT::Ptr& cloud, const Config& config) {
    if (!cloud || cloud->empty() || config.loop_icp_max_points <= 0 ||
        cloud->size() <= static_cast<std::size_t>(config.loop_icp_max_points)) {
        return cloud;
    }

    Keyframe::PointCloudT::Ptr prepared = cloud;
    const double voxel_size = std::max(
        config.loop_icp_prefilter_voxel_size,
        config.gicp_downsampling_resolution);
    if (voxel_size > 0.0) {
        Keyframe::PointCloudT::Ptr filtered;
        if (safeVoxelGridFilter<pcl::PointXYZI>(
                cloud, voxel_size, &filtered) &&
            filtered && !filtered->empty()) {
            prepared = filtered;
        }
    }
    return strideLimitCloud(
        prepared, static_cast<std::size_t>(config.loop_icp_max_points));
}

bool betterRegistration(
    const LoopVerification& candidate,
    const VisibilityConsistencyResult& candidate_visibility,
    const LoopVerification& incumbent,
    const VisibilityConsistencyResult& incumbent_visibility,
    bool prefer_visibility, bool has_incumbent) {
    if (!has_incumbent) return true;
    if (candidate.loop.verified != incumbent.loop.verified) {
        return candidate.loop.verified;
    }
    if (prefer_visibility) {
        if (candidate_visibility.valid != incumbent_visibility.valid) {
            return candidate_visibility.valid;
        }
        if (candidate_visibility.valid) {
            const double candidate_evidence =
                candidate_visibility.evidence_log_odds;
            const double incumbent_evidence =
                incumbent_visibility.evidence_log_odds;
            if (std::isfinite(candidate_evidence) !=
                std::isfinite(incumbent_evidence)) {
                return std::isfinite(candidate_evidence);
            }
            if (std::isfinite(candidate_evidence) &&
                std::abs(candidate_evidence - incumbent_evidence) > 1e-12) {
                return candidate_evidence > incumbent_evidence;
            }
        }
    }
    const double candidate_fitness = candidate.match_result.fitness_score;
    const double incumbent_fitness = incumbent.match_result.fitness_score;
    if (std::isfinite(candidate_fitness) !=
        std::isfinite(incumbent_fitness)) {
        return std::isfinite(candidate_fitness);
    }
    if (std::isfinite(candidate_fitness) &&
        std::abs(candidate_fitness - incumbent_fitness) > 1e-12) {
        return candidate_fitness < incumbent_fitness;
    }
    return candidate.match_result.inlier_ratio >
           incumbent.match_result.inlier_ratio;
}

}  // namespace

LoopVerificationPipeline::LoopVerificationPipeline(
    const Config& config, KeyframeManager& keyframe_manager,
    PointCloudMatcher& matcher, GraphOptimizer& optimizer,
    LoopClosureManager& loop_closure_manager)
    : config_(config),
      loop_verifier_(config_),
      consensus_verifier_(config_),
      keyframe_manager_(keyframe_manager),
      matcher_(matcher),
      optimizer_(optimizer),
      loop_closure_manager_(loop_closure_manager) {}

LoopVerificationPipelineResult LoopVerificationPipeline::evaluate(
    const LoopCandidate& candidate,
    const LoopVerificationContext& context) const {
    LoopVerificationPipelineResult result;
    result.loop.query_id = candidate.query_id;
    result.loop.match_id = candidate.match_id;

    const auto query = keyframe_manager_.getKeyframe(candidate.query_id);
    const auto match = keyframe_manager_.getKeyframe(candidate.match_id);
    if (!query || !match || !query->cloud || !match->cloud ||
        query->cloud->empty() || match->cloud->empty()) {
        result.reject_stage = "input";
        result.reject_reason = "missing_keyframe_or_cloud";
        return result;
    }
    if (context.cross_session &&
        (query->is_from_loaded_map || !match->is_from_loaded_map)) {
        result.reject_stage = "session_boundary";
        result.reject_reason = "candidate_not_loaded_map_to_new_session";
        return result;
    }

    const Eigen::Isometry3d predicted =
        match->pose_optimized.inverse() * query->pose_optimized;
    result.descriptor_seeded =
        context.cross_session && (candidate.fromRHPD() || candidate.fromSC());
    if (!result.descriptor_seeded &&
        predicted.translation().norm() > config_.loop_max_range) {
        result.reject_stage = "work_bound";
        result.reject_reason = "prediction_range_gate";
        return result;
    }

    result.source_registration_cloud =
        keyframe_manager_.buildSubmapInRootFrame(
            candidate.query_id, 0,
            result.descriptor_seeded ? candidate.query_id
                                     : candidate.match_id);
    result.target_registration_cloud =
        keyframe_manager_.buildSubmapInRootFrame(
            candidate.match_id, config_.gicp_submap_size,
            candidate.match_id, candidate.query_id);
    if (!result.source_registration_cloud ||
        result.source_registration_cloud->empty() ||
        !result.target_registration_cloud ||
        result.target_registration_cloud->empty()) {
        result.reject_stage = "submap";
        result.reject_reason = "empty_submap";
        return result;
    }

    result.source_registration_cloud =
        prepareLoopIcpCloud(result.source_registration_cloud, config_);
    result.target_registration_cloud =
        prepareLoopIcpCloud(result.target_registration_cloud, config_);
    if (!result.source_registration_cloud ||
        result.source_registration_cloud->size() < 10 ||
        !result.target_registration_cloud ||
        result.target_registration_cloud->size() < 10) {
        result.reject_stage = "prefilter";
        result.reject_reason = "empty_submap_after_prefilter";
        return result;
    }

    result.registration_attempted = true;
    const bool evaluate_visibility =
        static_cast<bool>(context.pose_visibility_evaluator);
    const auto pose_visibility = [&](const LoopVerification& verification) {
        if (!evaluate_visibility || !verification.loop.verified) {
            return VisibilityConsistencyResult{};
        }
        const Eigen::Isometry3d T_map_query =
            match->pose_optimized * verification.T_measured_match_query;
        return context.pose_visibility_evaluator(T_map_query);
    };
    if (result.descriptor_seeded) {
        const auto yaw_hypotheses =
            buildDescriptorYawHypotheses(candidate, config_);
        result.registration_hypothesis_count =
            static_cast<int>(yaw_hypotheses.size());
        const auto prepared_target =
            matcher_.prepareTargetCloud(result.target_registration_cloud);
        const auto prepared_source =
            matcher_.prepareSourceCloud(result.source_registration_cloud);
        bool has_verification = false;
        VisibilityConsistencyResult selected_visibility;
        for (double yaw : yaw_hypotheses) {
            Eigen::Isometry3d initial = Eigen::Isometry3d::Identity();
            initial.linear() = Eigen::AngleAxisd(
                yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            const auto registration =
                matcher_.alignPrepared(prepared_target, prepared_source, initial);
            auto verification = loop_verifier_.verifyQueryToMatchResult(
                candidate, query, match, result.source_registration_cloud,
                result.target_registration_cloud, registration);
            const auto visibility = pose_visibility(verification);
            if (betterRegistration(
                    verification, visibility, result.verification,
                    selected_visibility, evaluate_visibility,
                    has_verification)) {
                result.verification = std::move(verification);
                selected_visibility = visibility;
                result.selected_seed_yaw_rad = yaw;
                has_verification = true;
            }
        }
        result.pose_visibility = selected_visibility;
    } else {
        result.registration_hypothesis_count = 1;
        result.verification = loop_verifier_.verifyPreparedSubmaps(
            candidate, query, match, result.source_registration_cloud,
            result.target_registration_cloud, matcher_);
        result.pose_visibility = pose_visibility(result.verification);
    }
    result.pose_visibility_evaluated = evaluate_visibility;
    auto broad_loop = loop_closure_manager_.applyEdgeModel(result.verification.loop);
    if (broad_loop.verified) {
        assignLoopSegmentConsistency(&broad_loop,
            computeLoopSegmentConsistency(config_, keyframe_manager_, broad_loop));
        broad_loop.verified = LoopReferee::evaluate(makeSegmentAwareLoopFeatures(
            config_, candidate, broad_loop,
            result.verification.icp_translation_norm)).decision == LoopDecision::Accept;
    }
    if (broad_loop.verified && config_.icp_refine_use_gicp) {
        // Neighbor scans provide coverage for admission, but their estimated
        // relative poses can blur the target. Refine on the two original scans
        // only after that broad observation has passed, and require the refined
        // endpoint to retain the original submap's geometric support.
        const auto source = matcher_.prepareSourceCloud(query->cloud);
        const auto target = matcher_.prepareTargetCloud(match->cloud);
        const auto refined = refineConstraint(config_, matcher_, target, source,
            result.verification.T_measured_match_query);
        if (refined.success) {
            const auto submap = matcher_.prepareTargetCloud(result.target_registration_cloud);
            const auto supported = matcher_.evaluatePreparedPose(
                submap, source, refined.T_target_source, refined.metric);
            if (supported.num_inliers > 0 &&
                supported.fitness_score < config_.loop_fitness_threshold &&
                supported.inlier_ratio >= config_.loop_min_inlier_ratio) {
                result.verification = loop_verifier_.verifyQueryToMatchResult(
                    candidate, query, match, query->cloud, match->cloud, refined);
                result.source_registration_cloud = query->cloud;
                result.target_registration_cloud = match->cloud;
                result.pose_visibility = pose_visibility(result.verification);
            }
        }
    }
    result.loop = result.verification.loop;
    if (!result.loop.verified) {
        result.reject_stage = "registration";
        result.reject_reason = result.verification.reject_reason.empty()
                                   ? "registration_quality"
                                   : result.verification.reject_reason;
        return result;
    }
    if (evaluate_visibility &&
        (!result.pose_visibility.valid ||
         !std::isfinite(result.pose_visibility.evidence_log_odds) ||
         result.pose_visibility.evidence_log_odds <= 0.0)) {
        result.loop.verified = false;
        result.reject_stage = "visibility";
        result.reject_reason = "loaded_map_visibility_nonpositive";
        return result;
    }

    result.loop = loop_closure_manager_.applyEdgeModel(result.loop);
    if (!result.loop.verified) {
        result.reject_stage = "edge_model";
        result.reject_reason = "edge_model";
        return result;
    }

    const auto segment = computeLoopSegmentConsistency(
        config_, keyframe_manager_, result.loop);
    assignLoopSegmentConsistency(&result.loop, segment);
    const LoopFeatures features = makeSegmentAwareLoopFeatures(
        config_, candidate, result.loop,
        result.verification.icp_translation_norm);
    const LoopRefereeDecision referee = LoopReferee::evaluate(features);
    result.loop.loop_referee_energy = referee.energy;
    result.loop.loop_referee_recommendation =
        referee.decision == LoopDecision::Accept ? "accept" : "reject";
    result.loop.loop_referee_reason = referee.reason;
    result.loop.loop_referee_risk_flags = referee.risk_flags;
    result.loop.verified = referee.decision == LoopDecision::Accept;
    if (!result.loop.verified) {
        result.reject_stage = "loop_referee";
        result.reject_reason = "loop_referee";
    }
    return result;
}

}  // namespace n3mapping
