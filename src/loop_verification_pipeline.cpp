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

namespace n3mapping {
namespace {

Eigen::Vector3d rollPitchYaw(const Eigen::Isometry3d& transform) {
    return transform.rotation().eulerAngles(0, 1, 2);
}

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
        std::abs(rollPitchYaw(loop.T_icp_correction_match).z());
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

std::string consensusRejectReason(const VerifiedLoop& loop) {
    if (loop.consensus_estimator_recommendation ==
            "insufficient_estimator_support" &&
        loop.vertical_hypothesis_edge_recommendation == "planar_xy_yaw") {
        return "consensus_insufficient_planar";
    }

    constexpr double kLargeConsensusMeasurementDeltaM = 5.0;
    if (loop.consensus_estimator_recommendation ==
            "unstable_consensus_measurement" &&
        std::isfinite(loop.consensus_estimator_measurement_delta_translation) &&
        loop.consensus_estimator_measurement_delta_translation >=
            kLargeConsensusMeasurementDeltaM) {
        return "consensus_unstable_large_delta";
    }
    return {};
}

bool graphTrialYawInconsistent(
    const LoopGraphTrialDiagnostics& diagnostics) {
    constexpr double kConstraintRotationDegradationRad = 0.05;
    if (!diagnostics.success) {
        return false;
    }
    return std::isfinite(diagnostics.existing_loop_residual_delta) &&
           diagnostics.existing_loop_residual_delta >
               kConstraintRotationDegradationRad;
}

bool graphTrialTranslationInconsistent(
    const LoopGraphTrialDiagnostics& diagnostics) {
    constexpr double kConstraintDegradationM = 0.05;
    if (!diagnostics.success) {
        return false;
    }
    return (std::isfinite(diagnostics.existing_loop_residual_delta) &&
            diagnostics.existing_loop_residual_delta >
                kConstraintDegradationM) ||
           (std::isfinite(diagnostics.odom_residual_delta) &&
            diagnostics.odom_residual_delta > kConstraintDegradationM);
}

void assignGraphTrialDiagnostics(
    VerifiedLoop* loop, const LoopGraphTrialDiagnostics& diagnostics) {
    if (!loop) {
        return;
    }
    loop->graph_trial_success = diagnostics.success;
    loop->graph_trial_residual_x_after = diagnostics.residual_x_after;
    loop->graph_trial_residual_y_after = diagnostics.residual_y_after;
    loop->graph_trial_residual_z_after = diagnostics.residual_z_after;
    loop->graph_trial_residual_roll_after = diagnostics.residual_roll_after;
    loop->graph_trial_residual_pitch_after = diagnostics.residual_pitch_after;
    loop->graph_trial_residual_yaw_after = diagnostics.residual_yaw_after;
    loop->graph_trial_residual_translation_norm_after =
        diagnostics.residual_translation_norm_after;
    loop->graph_trial_residual_rotation_norm_after =
        diagnostics.residual_rotation_norm_after;
    loop->graph_trial_mean_pose_update_translation =
        diagnostics.mean_pose_update_translation;
    loop->graph_trial_max_pose_update_translation =
        diagnostics.max_pose_update_translation;
    loop->graph_trial_mean_pose_update_rotation =
        diagnostics.mean_pose_update_rotation;
    loop->graph_trial_max_pose_update_rotation =
        diagnostics.max_pose_update_rotation;
    loop->graph_trial_existing_loop_residual_delta =
        diagnostics.existing_loop_residual_delta;
    loop->graph_trial_odom_residual_delta = diagnostics.odom_residual_delta;
    loop->graph_trial_consistency_score = diagnostics.consistency_score;
    loop->graph_trial_recommendation = diagnostics.recommendation;
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
            candidate.match_id);
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
        bool has_verification = false;
        VisibilityConsistencyResult selected_visibility;
        for (double yaw : yaw_hypotheses) {
            Eigen::Isometry3d initial = Eigen::Isometry3d::Identity();
            initial.linear() = Eigen::AngleAxisd(
                yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            auto verification = loop_verifier_.verifyPreparedQueryToMatch(
                candidate, query, match, result.source_registration_cloud,
                result.target_registration_cloud, initial, matcher_);
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

LoopConstraintPipelineResult LoopVerificationPipeline::evaluateConstraint(
    const VerifiedLoop& input_loop, LoopEdgeDirection direction,
    const LoopConstraintContext& context) const {
    LoopConstraintPipelineResult result;
    result.loop = input_loop;
    if (!result.loop.verified) {
        result.reject_stage = "selection";
        result.reject_reason = "loop_not_verified";
        return result;
    }

    const auto query = keyframe_manager_.getKeyframe(result.loop.query_id);
    const auto match = keyframe_manager_.getKeyframe(result.loop.match_id);
    if (!query || !match) {
        result.reject_stage = "input";
        result.reject_reason = "missing_keyframe";
        return result;
    }
    if (context.cross_session &&
        (query->is_from_loaded_map || !match->is_from_loaded_map)) {
        result.reject_stage = "session_boundary";
        result.reject_reason = "constraint_not_loaded_map_to_new_session";
        return result;
    }

    result.consensus = consensus_verifier_.evaluate(
        keyframe_manager_, matcher_, result.loop,
        std::max(2, config_.gicp_submap_size), context.cross_session);
    assignLoopConsensus(&result.loop, result.consensus);
    if (context.cross_session &&
        result.consensus.decision != LoopConsensusDecision::Commit) {
        result.reject_stage = "consensus";
        result.reject_reason =
            result.consensus.decision == LoopConsensusDecision::Reject
                ? (result.consensus.reason.empty()
                       ? "neighborhood_contradiction"
                       : result.consensus.reason)
                : "session_merge_consensus_deferred";
        return result;
    }
    if (const std::string reason = consensusRejectReason(result.loop);
        !reason.empty()) {
        result.reject_stage = "consensus";
        result.reject_reason = reason;
        return result;
    }

    const auto edges = loop_closure_manager_.buildLoopEdges(
        {result.loop}, direction);
    if (edges.empty()) {
        result.reject_stage = "edge_build";
        result.reject_reason = "edge_build_empty";
        return result;
    }
    result.edge = edges.front();
    result.has_edge = true;

    const auto poses_before = optimizer_.getOptimizedPoses();
    const auto committed_edges = optimizer_.getEdges();
    result.graph_trial = computeLoopGraphTrialDiagnostics(
        config_, poses_before, committed_edges, {result.edge});
    assignGraphTrialDiagnostics(&result.loop, result.graph_trial);

    if (result.consensus.estimator_pair_count >= 3) {
        EdgeInfo estimator_edge = result.edge;
        estimator_edge.measurement =
            result.consensus.estimator_measurement_match_query;
        result.consensus_estimator_trial = computeLoopGraphTrialDiagnostics(
            config_, poses_before, committed_edges, {estimator_edge});
        result.has_consensus_estimator_trial = true;
    }

    if (context.cross_session && !result.graph_trial.success) {
        result.reject_stage = "graph_trial";
        result.reject_reason = "graph_trial_unavailable";
        return result;
    }
    if (graphTrialYawInconsistent(result.graph_trial)) {
        result.reject_stage = "graph_trial";
        result.reject_reason = "graph_inconsistent_yaw";
        return result;
    }
    if (graphTrialTranslationInconsistent(result.graph_trial)) {
        result.reject_stage = "graph_trial";
        result.reject_reason = "graph_inconsistent_translation";
        return result;
    }

    result.accepted = true;
    return result;
}

}  // namespace n3mapping
