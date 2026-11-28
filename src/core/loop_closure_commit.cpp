#include "n3mapping/graph_optimizer.h"

#include "loop_closure_processing.h"
#include "loop_closure_diagnostics.h"
#include "n3mapping/loop_graph_trial_diagnostics.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_verification_pipeline.h"

#include <map>

namespace n3mapping {
using namespace loop_diagnostics;

void recordLoopOptimization(const Config& config, const std::vector<EdgeInfo>& edges,
    const std::vector<VerifiedLoop>& best_loops,
    const std::map<int64_t, Eigen::Isometry3d>& poses_before,
    const std::map<int64_t, Eigen::Isometry3d>& poses_after,
    LoopClosureOutcome& outcome) {
  auto& result = outcome.summary;
  const bool loop_debug_enabled = config.loop_debug_enable;
  const auto residual_before = meanLoopResidual(edges, poses_before, loop_debug_enabled);
  const auto residual_after = meanLoopResidual(edges, poses_after, loop_debug_enabled);
  result.optimized = true;
  result.edge_count += edges.size();
  result.graph_edge_count += edges.size();
  result.loop_residual_translation_before = residual_before.translation_norm;
  result.loop_residual_rotation_before = residual_before.rotation_angle;
  result.loop_residual_translation_after = residual_after.translation_norm;
  result.loop_residual_rotation_after = residual_after.rotation_angle;
  accumulatePoseUpdateStats(poses_before, poses_after, result);
  result.accepted_loops.insert(result.accepted_loops.end(),
                               best_loops.begin(), best_loops.end());

  if (loop_debug_enabled) {
    LoopDebugOptimizationEvent event;
    event.processing_time = processingTimeSeconds();
    event.accepted_edge_count = edges.size();
    event.accepted_edges.reserve(edges.size());
    for (const auto &edge : edges) {
      event.accepted_edges.emplace_back(edge.from_id, edge.to_id);
    }
    event.loop_residual_translation_before = residual_before.translation_norm;
    event.loop_residual_rotation_before = residual_before.rotation_angle;
    event.loop_residual_translation_after = residual_after.translation_norm;
    event.loop_residual_rotation_after = residual_after.rotation_angle;
    event.loop_residual_translation_axes_before =
        residual_before.translation;
    event.loop_residual_translation_axes_after =
        residual_after.translation;
    event.loop_residual_rpy_axes_before = residual_before.rpy;
    event.loop_residual_rpy_axes_after = residual_after.rpy;
    event.mean_pose_update_translation = result.mean_pose_update_translation;
    event.max_pose_update_translation = result.max_pose_update_translation;
    event.mean_pose_update_rotation = result.mean_pose_update_rotation;
    event.max_pose_update_rotation = result.max_pose_update_rotation;
    outcome.optimizations.push_back(event);
  }
}
void rebaseLoopOutcome(const Config& config, GraphOptimizer& optimizer,
                       const KeyframeManager& keyframes, LoopClosureManager& manager,
                       LoopClosureOutcome& outcome, const LoopAnchorOffsets& anchor_offsets) {

  std::map<int64_t, std::vector<VerifiedLoop>> by_query;
  for (const auto& loop : outcome.summary.accepted_loops)
    by_query[loop.query_id].push_back(loop);
  outcome.summary = {};
  outcome.optimizations.clear();
  LoopDriftBudget budget;
  if (!optimizer.fixedNodes().empty())
    budget.paths = measureLoopOdometryPaths(keyframes.getAllKeyframes(), anchor_offsets);

  for (auto& [query_id, loops] : by_query) {
    bool accepted_anchor = false;
    loops.erase(std::remove_if(loops.begin(), loops.end(), [&](const auto& loop) {
      if (!optimizer.fixedNodes().count(loop.match_id) ||
          optimizer.fixedNodes().count(query_id)) return false;
      const double untracked = budget.untracked(*keyframes.getKeyframe(query_id));
      if (withinLoopDriftEnvelope(optimizer.getOptimizedPose(loop.match_id),
            optimizer.getOptimizedPose(query_id), loop.T_measured_match_query,
            maxHorizontalDrift(untracked, config.reloc_search_radius),
            maxVerticalDrift(untracked))) {
        accepted_anchor = true;
        return false;
      }
      for (auto& event : outcome.candidates) {
        if (event.candidate.query_id == query_id && event.candidate.match_id == loop.match_id &&
            event.gate_result == "accepted") {
          event.gate_result = "rejected";
          event.reject_reason = "async_rebase_drift_envelope_gate";
        }
      }
      return true;
    }), loops.end());
    if (loops.empty()) continue;
    const auto edges = manager.buildLoopEdges(loops, LoopEdgeDirection::MatchToQuery);
    const auto before = optimizer.getOptimizedPoses();
    std::string rejection;
    for (const auto& edge : edges) {
      const auto trial = computeLoopGraphTrialDiagnostics(config, optimizer, {edge});
      const bool cross_session = optimizer.fixedNodes().count(edge.from_id) &&
          !optimizer.fixedNodes().count(edge.to_id);
      for (auto& loop : loops)
        if (loop.match_id == edge.from_id && loop.query_id == edge.to_id)
          assignLoopGraphTrialDiagnostics(&loop, trial);
      for (auto& event : outcome.candidates)
        if (event.candidate.match_id == edge.from_id && event.candidate.query_id == edge.to_id)
          assignGraphTrialDiagnostics(event, trial);
      const auto reason = loopGraphTrialRejectReason(trial, cross_session);
      if (!reason.empty()) rejection = reason;
    }
    if (rejection.empty() && optimizer.optimizeLoopEdges(edges)) {
      if (accepted_anchor) budget.anchor(*keyframes.getKeyframe(query_id));
      outcome.summary.place_candidate_count += loops.size();
      recordLoopOptimization(config, edges, loops, before, optimizer.getOptimizedPoses(), outcome);
      continue;
    }
    for (auto& event : outcome.candidates) {
      if (event.candidate.query_id != query_id || event.gate_result != "accepted") continue;
      event.gate_result = "rejected";
      event.reject_reason = "async_rebase_" + (rejection.empty() ? "optimization_failed" : rejection);
    }
  }
}

}
