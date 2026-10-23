// Commit retained geometric measurements and rebase them without rerunning ICP.
#include "loop_closure_processing.h"
#include "loop_closure_diagnostics.h"
#include "n3mapping/loop_graph_trial_diagnostics.h"

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
  const auto residual_before = meanLoopResidual(edges, poses_before);
  const auto residual_axes_before = meanLoopResidualAxes(edges, poses_before);
  const auto residual_after = meanLoopResidual(edges, poses_after);
  const auto residual_axes_after = meanLoopResidualAxes(edges, poses_after);
  result.optimized = true;
  result.edge_count += edges.size();
  result.graph_edge_count += edges.size();
  result.loop_residual_translation_before = residual_before.first;
  result.loop_residual_rotation_before = residual_before.second;
  result.loop_residual_translation_after = residual_after.first;
  result.loop_residual_rotation_after = residual_after.second;
  accumulatePoseUpdateStats(poses_before, poses_after, &result);
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
    event.loop_residual_translation_before = residual_before.first;
    event.loop_residual_rotation_before = residual_before.second;
    event.loop_residual_translation_after = residual_after.first;
    event.loop_residual_rotation_after = residual_after.second;
    event.loop_residual_translation_axes_before =
        residual_axes_before.translation;
    event.loop_residual_translation_axes_after =
        residual_axes_after.translation;
    event.loop_residual_rpy_axes_before = residual_axes_before.rpy;
    event.loop_residual_rpy_axes_after = residual_axes_after.rpy;
    event.mean_pose_update_translation = result.mean_pose_update_translation;
    event.max_pose_update_translation = result.max_pose_update_translation;
    event.mean_pose_update_rotation = result.mean_pose_update_rotation;
    event.max_pose_update_rotation = result.max_pose_update_rotation;
    outcome.optimizations.push_back(event);
  }
}
void rebaseLoopOutcome(const Config& config, GraphOptimizer& optimizer,
                       LoopClosureManager& manager, LoopClosureOutcome& outcome) {
  // Geometry is an independent relative measurement. Recheck graph admission
  // against all updates made while the worker was busy; never rerun ICP here.
  std::map<int64_t, std::vector<VerifiedLoop>> by_query;
  for (const auto& loop : outcome.summary.accepted_loops)
    by_query[loop.query_id].push_back(loop);
  outcome.summary = {};
  outcome.optimizations.clear();
  // Rebase each retained same-query consensus separately. A rejected query must
  // not discard unrelated verified measurements that shared its worker batch.
  for (auto& [query_id, loops] : by_query) {
    const auto edges = manager.buildLoopEdges(loops, LoopEdgeDirection::MatchToQuery);
    const auto before = optimizer.getOptimizedPoses();
    const auto committed = optimizer.getEdges();
    std::string rejection;
    for (const auto& edge : edges) {
      const auto trial = computeLoopGraphTrialDiagnostics(config, before, committed, {edge});
      for (auto& loop : loops)
        if (loop.match_id == edge.from_id && loop.query_id == edge.to_id)
          assignLoopGraphTrialDiagnostics(&loop, trial);
      for (auto& event : outcome.candidates)
        if (event.candidate.match_id == edge.from_id && event.candidate.query_id == edge.to_id)
          assignGraphTrialDiagnostics(&event, trial);
      const auto reason = loopGraphTrialRejectReason(trial);
      if (!reason.empty()) rejection = reason;
    }
    if (rejection.empty() && !edges.empty() && manager.applyEdges(edges, optimizer)) {
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

} // namespace n3mapping
