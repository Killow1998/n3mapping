// Shared graph admission for mapping and map extension. No graph mutation
// occurs until every candidate has its own neighborhood and trial evidence.
#include "n3mapping/loop_verification_pipeline.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <Eigen/Cholesky>
#include <gtsam/geometry/Pose3.h>
#include "n3mapping/n3map_proto_utils.h"

namespace n3mapping {
namespace {
void accountForNeighborDisagreement(
    const Config& config, const LoopConsensusResult& consensus,
    VerifiedLoop* loop) {
    if (!config.loop_use_icp_information ||
        loop->edge_mode != LoopEdgeMode::Full6Dof) return;
    using Matrix6 = Eigen::Matrix<double, 6, 6>;
    Matrix6 scatter = Matrix6::Zero();
    int count = 0;
    for (const auto& pair : consensus.pairs) {
        if (!pair.valid || !pair.converged || !pair.has_estimated_measurement ||
            !std::isfinite(pair.fitness_score) || !std::isfinite(pair.inlier_ratio) ||
            pair.fitness_score >= config.loop_fitness_threshold ||
            pair.inlier_ratio < config.loop_min_inlier_ratio) continue;
        const Eigen::Isometry3d delta =
            loop->T_measured_match_query.inverse() * pair.estimated_match_query;
        const auto tangent = gtsam::Pose3::Logmap(gtsam::Pose3(delta.matrix()));
        Eigen::Matrix<double, 6, 1> residual;
        residual << tangent.tail<3>(), tangent.head<3>();
        if (!residual.allFinite()) continue;
        scatter.noalias() += residual * residual.transpose();
        ++count;
    }
    if (count < 3) return;
    // Neighbor estimates share scans and odometry. Their second moment is an
    // uncertainty floor, not independent information to multiply by their count.
    const Eigen::LDLT<Matrix6> initial(loop->information);
    if (initial.info() != Eigen::Success || !initial.isPositive()) return;
    const Matrix6 covariance = initial.solve(Matrix6::Identity()) + scatter / count;
    const Eigen::LDLT<Matrix6> calibrated(covariance);
    if (calibrated.info() != Eigen::Success || !calibrated.isPositive()) return;
    const Matrix6 candidate = calibrated.solve(Matrix6::Identity());
    const Matrix6 information = 0.5 * (candidate + candidate.transpose());
    std::string error;
    if (isValidInformationMatrix(information, &error))
        loop->information = information;
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


}  // namespace

std::vector<LoopConstraintPipelineResult>
LoopVerificationPipeline::evaluateConstraints(
    const std::vector<VerifiedLoop>& input_loops, LoopEdgeDirection direction,
    const LoopConstraintContext& context) const {
    std::vector<LoopConstraintPipelineResult> results(input_loops.size());
    std::vector<VerifiedLoop> eligible;
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < input_loops.size(); ++i) {
        auto& result = results[i];
        result.loop = input_loops[i];
        if (!result.loop.verified) {
            result.reject_stage = "selection";
            result.reject_reason = "loop_not_verified";
            continue;
        }

        const auto query = keyframe_manager_.getKeyframe(result.loop.query_id);
        const auto match = keyframe_manager_.getKeyframe(result.loop.match_id);
        if (!query || !match) {
            result.reject_stage = "input";
            result.reject_reason = "missing_keyframe";
            continue;
        }
        if (context.cross_session &&
            (query->is_from_loaded_map || !match->is_from_loaded_map)) {
            result.reject_stage = "session_boundary";
            result.reject_reason = "constraint_not_loaded_map_to_new_session";
            continue;
        }

        eligible.push_back(result.loop);
        indices.push_back(i);
    }
    auto evidence = consensus_verifier_.evaluate(
        keyframe_manager_, matcher_, eligible,
        std::max(2, config_.gicp_submap_size), context.cross_session);
    for (std::size_t i = 0; i < eligible.size(); ++i) {
        auto& result = results[indices[i]];
        result.consensus = std::move(evidence[i]);
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
            continue;
        }
        if (const std::string reason = consensusRejectReason(result.loop);
            !reason.empty()) {
            result.reject_stage = "consensus";
            result.reject_reason = reason;
            continue;
        }
        accountForNeighborDisagreement(config_, result.consensus, &result.loop);

        const auto edges = loop_closure_manager_.buildLoopEdges(
            {result.loop}, direction);
        if (edges.empty()) {
            result.reject_stage = "edge_build";
            result.reject_reason = "edge_build_empty";
            continue;
        }
        result.edge = edges.front();
        result.has_edge = true;

        const auto poses_before = optimizer_.getOptimizedPoses();
        const auto committed_edges = optimizer_.getEdges();
        result.graph_trial = computeLoopGraphTrialDiagnostics(
            config_, poses_before, committed_edges, {result.edge});
        assignLoopGraphTrialDiagnostics(&result.loop, result.graph_trial);

        // This alternative measurement trial is diagnostic only. The actual
        // candidate trial above remains mandatory for every constraint decision.
        // Only the mapping debug stream consumes the alternative result.
        if (config_.loop_debug_enable && !context.cross_session &&
            result.consensus.estimator_pair_count >= 3) {
            EdgeInfo estimator_edge = result.edge;
            estimator_edge.measurement =
                result.consensus.estimator_measurement_match_query;
            result.consensus_estimator_trial = computeLoopGraphTrialDiagnostics(
                config_, poses_before, committed_edges, {estimator_edge});
            result.has_consensus_estimator_trial = true;
        }

        if (const auto reason = loopGraphTrialRejectReason(
                result.graph_trial, context.cross_session); !reason.empty()) {
            result.reject_stage = "graph_trial";
            result.reject_reason = reason;
            continue;
        }

        result.accepted = true;
    }
    return results;
}

}  // namespace n3mapping
