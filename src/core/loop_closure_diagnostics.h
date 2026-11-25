#pragma once
#include "loop_closure_processing.h"
#include "n3mapping/core/types.h"
#include "n3mapping/loop_consensus_verifier.h"
#include "n3mapping/loop_graph_trial_diagnostics.h"
namespace n3mapping { struct LoopVerificationPipelineResult; }

namespace n3mapping::loop_diagnostics {
struct LoopResidualStats {
  double translation_norm = 0.0;
  double rotation_angle = 0.0;
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Vector3d rpy = Eigen::Vector3d::Zero();
};

double rotationAngle(const Eigen::Isometry3d& transform);
LoopResidualStats meanLoopResidual(const std::vector<EdgeInfo>& edges,
    const std::map<int64_t, Eigen::Isometry3d>& poses, bool include_axes = false);
void accumulatePoseUpdateStats(const std::map<int64_t, Eigen::Isometry3d>& before,
    const std::map<int64_t, Eigen::Isometry3d>& after, CoreLoopClosureResult& result);
void assignGraphTrialDiagnostics(LoopDebugCandidateEvent& event,
    const LoopGraphTrialDiagnostics& diagnostics);
void assignConsensusEstimatorTrialDiagnostics(LoopDebugCandidateEvent& event,
    const LoopGraphTrialDiagnostics& diagnostics);
void assignSegmentDiagnostics(LoopDebugCandidateEvent& event, const VerifiedLoop& loop);
void assignConsensusDiagnostics(LoopDebugCandidateEvent& event, const VerifiedLoop& loop);
void assignConsensusDiagnostics(LoopDebugCandidateEvent& event, const LoopConsensusResult& consensus);
double processingTimeSeconds();
LoopDebugCandidateEvent makeCandidateDebugEvent(const LoopCandidate& candidate, double timestamp,
    LoopVerificationPipelineResult& result, PointCloudMatcher& matcher, bool vertical_hypotheses);
}
