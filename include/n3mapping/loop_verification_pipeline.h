#pragma once

#include <functional>
#include <limits>
#include <string>

#include "n3mapping/config.h"
#include "n3mapping/graph_types.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_closure_manager.h"
#include "n3mapping/loop_consensus_verifier.h"
#include "n3mapping/loop_graph_trial_diagnostics.h"
#include "n3mapping/loop_verifier.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/visibility_consistency.h"

namespace n3mapping {
class GraphOptimizer;

struct LoopVerificationContext {
    bool cross_session = false;
    std::function<VisibilityConsistencyResult(const Eigen::Isometry3d&)>
        pose_visibility_evaluator;
    double max_horizontal_drift = 0.0;
    double max_vertical_drift = 0.0;
};

inline bool withinLoopDriftEnvelope(const Eigen::Isometry3d& match_to_map,
                                   const Eigen::Isometry3d& query_to_map,
                                   const Eigen::Isometry3d& measured_query_to_match,
                                   double horizontal, double vertical) {
    const Eigen::Vector3d delta =
        (match_to_map * measured_query_to_match).translation() - query_to_map.translation();
    return (vertical <= 0.0 || std::abs(delta.z()) <= vertical) &&
           (horizontal <= 0.0 || delta.head<2>().norm() <= horizontal);
}

struct LoopVerificationPipelineResult {
    LoopVerification verification;
    VerifiedLoop loop;
    Keyframe::PointCloudT::Ptr source_registration_cloud;
    Keyframe::PointCloudT::Ptr target_registration_cloud;
    bool registration_attempted = false;
    bool descriptor_seeded = false;
    int registration_hypothesis_count = 0;
    double selected_seed_yaw_rad =
        std::numeric_limits<double>::quiet_NaN();
    bool pose_visibility_evaluated = false;
    VisibilityConsistencyResult pose_visibility;
    std::string reject_stage;
    std::string reject_reason;
};

struct LoopConstraintContext {
    bool cross_session = false;
};

struct LoopConstraintPipelineResult {
    VerifiedLoop loop;
    LoopConsensusResult consensus;
    LoopGraphTrialDiagnostics graph_trial;
    LoopGraphTrialDiagnostics consensus_estimator_trial;
    EdgeInfo edge;
    bool has_edge = false;
    bool has_consensus_estimator_trial = false;
    bool accepted = false;
    std::string reject_stage;
    std::string reject_reason;
};

class LoopVerificationPipeline {
public:
    LoopVerificationPipeline(const Config& config,
                             KeyframeManager& keyframe_manager,
                             PointCloudMatcher& matcher,
                             GraphOptimizer& optimizer,
                             LoopClosureManager& loop_closure_manager);

    LoopVerificationPipelineResult evaluate(
        const LoopCandidate& candidate,
        const LoopVerificationContext& context = {}) const;

    std::vector<LoopConstraintPipelineResult> evaluateConstraints(
        const std::vector<VerifiedLoop>& loops,
        LoopEdgeDirection direction,
        const LoopConstraintContext& context = {}) const;

private:
    Config config_;
    LoopVerifier loop_verifier_;
    LoopConsensusVerifier consensus_verifier_;
    KeyframeManager& keyframe_manager_;
    PointCloudMatcher& matcher_;
    GraphOptimizer& optimizer_;
    LoopClosureManager& loop_closure_manager_;
};

}
