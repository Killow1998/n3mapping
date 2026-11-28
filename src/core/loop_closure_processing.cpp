#include "loop_closure_processing.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "loop_closure_diagnostics.h"

#include <array>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <glog/logging.h>

#include "n3mapping/loop_graph_trial_diagnostics.h"
#include "n3mapping/loop_verification_pipeline.h"

namespace n3mapping {
using namespace loop_diagnostics;

LoopOdometryPaths measureLoopOdometryPaths(
    const std::vector<Keyframe::Ptr>& keyframes,
    const std::unordered_map<int64_t, double>& anchor_offsets) {
  struct SessionPath {
    const Keyframe* previous = nullptr;
    double length = 0.0;
    double anchor_length = 0.0;
  };
  std::unordered_map<MapSessionId, SessionPath> sessions;
  LoopOdometryPaths paths;
  for (const auto& keyframe : keyframes) {
    if (keyframe->is_from_loaded_map) continue;
    auto& session = sessions[keyframe->session_id];
    if (session.previous) {
      session.length += (keyframe->pose_odom.translation() -
                         session.previous->pose_odom.translation()).norm();
    }
    const auto anchor = anchor_offsets.find(keyframe->id);
    if (anchor != anchor_offsets.end()) session.anchor_length = session.length - anchor->second;
    paths.emplace(keyframe->id, LoopOdometryPath{session.length, session.length - session.anchor_length});
    session.previous = keyframe.get();
  }
  return paths;
}

LoopClosureOutcome processLoopCandidates(
    const Config& config, KeyframeManager& keyframes, PointCloudMatcher& matcher,
    GraphOptimizer& optimizer, LoopClosureManager& manager,
    const std::vector<DetectedLoopQuery>& queries,
    const LoadedMapVisibilityEvaluator& visibility) {
  LoopClosureOutcome outcome;
  auto& result = outcome.summary;
  LoopVerificationPipeline verification_pipeline(config, keyframes, matcher, optimizer, manager);
  std::vector<EdgeInfo> batch_edges;
  std::vector<VerifiedLoop> batch_loops;
  for (const auto& query : queries) {
    const auto query_kf = keyframes.getKeyframe(query.id);
    if (!query_kf) continue;
    const auto& candidates = query.candidates;
    std::array<std::vector<VerifiedLoop>, 2> verified_groups;
    std::vector<LoopDebugCandidateEvent> debug_events;
    std::map<std::pair<int64_t, int64_t>, EdgeInfo> edge_by_pair;
    std::map<std::pair<int64_t, int64_t>, std::string> graph_reject_by_pair;
    const bool loop_debug_enabled = config.loop_debug_enable;
    if (loop_debug_enabled) {
      debug_events.reserve(candidates.size());
    }

    auto flush_debug_events =
        [&](const std::set<std::pair<int64_t, int64_t>> &accepted_pairs,
            const std::string &not_selected_reason) {
          if (VLOG_IS_ON(1)) {
            for (const auto& candidate : candidates) {
              const auto key = std::make_pair(candidate.query_id, candidate.match_id);
              const auto rejected = graph_reject_by_pair.find(key);
              VLOG(1) << "[Loop/Selection] query=" << candidate.query_id
                      << " match=" << candidate.match_id
                      << " accepted=" << accepted_pairs.count(key)
                      << " reason=" << (accepted_pairs.count(key) ? "accepted" :
                          rejected != graph_reject_by_pair.end() ? rejected->second :
                          not_selected_reason.empty() ? "not_selected" : not_selected_reason);
            }
          }
          if (!loop_debug_enabled) {
            return;
          }
          for (auto &event : debug_events) {
            const std::pair<int64_t, int64_t> key(event.candidate.query_id,
                                                  event.candidate.match_id);
            if (accepted_pairs.find(key) != accepted_pairs.end()) {
              event.gate_result = "accepted";
              event.reject_reason.clear();
            } else if (event.gate_result == "accepted") {
              event.gate_result = "rejected";
              auto graph_reject_it = graph_reject_by_pair.find(key);
              if (graph_reject_it != graph_reject_by_pair.end()) {
                event.reject_reason = graph_reject_it->second;
              } else {
                event.reject_reason = not_selected_reason.empty()
                                          ? "not_selected"
                                          : not_selected_reason;
              }
            }
            outcome.candidates.push_back(std::move(event));
          }
        };

    for (const auto &candidate : candidates) {
      const auto match = keyframes.getKeyframe(candidate.match_id);
      if (!match) continue;
      const bool cross_session = !query_kf->is_from_loaded_map && match->is_from_loaded_map;
      LoopVerificationContext context;
      context.cross_session = cross_session;
      if (cross_session) {
        context.max_horizontal_drift = query.max_horizontal_drift;
        context.max_vertical_drift = query.max_vertical_drift;
        context.pose_visibility_evaluator = [&](const Eigen::Isometry3d& pose) {
          return visibility ? visibility(*query_kf->cloud, pose)
                            : VisibilityConsistencyResult{};
        };
      }
      auto pipeline_result = verification_pipeline.evaluate(candidate, context);
      VLOG(1) << "[Loop/Verification] query=" << candidate.query_id
              << " match=" << candidate.match_id
              << " registered=" << pipeline_result.registration_attempted
              << " verified=" << pipeline_result.loop.verified
              << " fitness=" << pipeline_result.verification.match_result.fitness_score
              << " inlier_ratio=" << pipeline_result.verification.match_result.inlier_ratio
              << " stage=" << pipeline_result.reject_stage
              << " reason=" << pipeline_result.reject_reason
              << " referee=" << pipeline_result.loop.loop_referee_reason
              << " segment_pairs=" << pipeline_result.loop.segment_valid_pair_count
              << " segment_ratio=" << pipeline_result.loop.segment_consensus_ratio;
      if (loop_debug_enabled) {
        debug_events.push_back(makeCandidateDebugEvent(candidate, query_kf->timestamp,
            pipeline_result, matcher, config.loop_debug_vertical_hypotheses_enable));
      }
      if (pipeline_result.registration_attempted)
        verified_groups[cross_session].push_back(std::move(pipeline_result.loop));
    }

    std::vector<VerifiedLoop> gated_loops;

    std::multimap<std::pair<int64_t, int64_t>, LoopDebugCandidateEvent*> events_by_pair;
    for (auto& event : debug_events)
      events_by_pair.emplace(std::make_pair(event.candidate.query_id, event.candidate.match_id), &event);

    for (size_t group = 0; group < verified_groups.size(); ++group) {
      const auto valid = manager.filterValidLoops(verified_groups[group]);
      const auto selected = config.loop_keep_all_verified ? valid : manager.selectBestPerQuery(valid);
      for (const auto& gate : verification_pipeline.evaluateConstraints(
               selected, LoopEdgeDirection::MatchToQuery, {group != 0})) {
        const auto key = std::make_pair(gate.loop.query_id, gate.loop.match_id);
        if (gate.has_edge) edge_by_pair[key] = gate.edge;
        const auto events = events_by_pair.equal_range(key);
        for (auto it = events.first; it != events.second; ++it) {
          auto& event = *it->second;
          assignConsensusDiagnostics(event, gate.consensus);
          event.loop_referee_recommendation = gate.loop.loop_referee_recommendation;
          event.loop_referee_reason = gate.loop.loop_referee_reason;
          event.loop_referee_risk_flags = gate.loop.loop_referee_risk_flags;
          if (gate.has_edge) {
            event.loop_information = gate.edge.information;
            assignGraphTrialDiagnostics(event, gate.graph_trial);
          }
          if (gate.has_consensus_estimator_trial)
            assignConsensusEstimatorTrialDiagnostics(event, gate.consensus_estimator_trial);
        }
        if (!gate.accepted || !gate.has_edge) {
          graph_reject_by_pair[key] = gate.reject_reason.empty()
                                          ? "verification_pipeline_rejected"
                                          : gate.reject_reason;
          continue;
        }
        gated_loops.push_back(gate.loop);
      }
    }

    if (gated_loops.empty()) {
      flush_debug_events({}, "verification_pipeline_rejected");
      continue;
    }

    std::vector<VerifiedLoop> best_loops;
    if (config.loop_keep_all_verified) {
      std::map<int64_t, Keyframe::Ptr> keyframe_map;
      for (const auto& loop : gated_loops) {
        keyframe_map.emplace(loop.query_id, keyframes.getKeyframe(loop.query_id));
        keyframe_map.emplace(loop.match_id, keyframes.getKeyframe(loop.match_id));
      }
      auto selection = manager.selectSameQueryConsensus(
          gated_loops, keyframe_map);
      best_loops = std::move(selection.selected);
      graph_reject_by_pair.insert(selection.rejected.begin(),
                                  selection.rejected.end());
    } else {
      best_loops = std::move(gated_loops);
    }

    std::vector<EdgeInfo> edges;
    edges.reserve(best_loops.size());
    for (const auto& loop : best_loops)
      edges.push_back(edge_by_pair.at({loop.query_id, loop.match_id}));
    if (edges.empty()) {
      flush_debug_events({}, "same_query_consensus_rejected");
      continue;
    }
    if (config.loop_geometry_enable && queries.size() > 1 && optimizer.fixedNodes().empty()) {
      batch_edges.insert(batch_edges.end(), edges.begin(), edges.end());
      batch_loops.insert(batch_loops.end(), best_loops.begin(), best_loops.end());
      std::set<std::pair<int64_t, int64_t>> pairs;
      for (const auto& loop : best_loops) pairs.emplace(loop.query_id, loop.match_id);
      flush_debug_events(pairs, "not_selected");
      continue;
    }
    const auto poses_before = optimizer.getOptimizedPoses();
    result.place_candidate_count += best_loops.size();
    const bool optimization_committed =
        optimizer.optimizeLoopEdges(edges);
    if (!optimization_committed) {
      flush_debug_events({}, "optimization_failed");
      continue;
    }

    std::set<std::pair<int64_t, int64_t>> accepted_debug_pairs;
    for (const auto &loop : best_loops) {
      accepted_debug_pairs.insert({loop.query_id, loop.match_id});
    }
    flush_debug_events(accepted_debug_pairs, "not_selected");

    const auto poses_after = optimizer.getOptimizedPoses();
    keyframes.updateOptimizedPoses(poses_after);

    recordLoopOptimization(config, edges, best_loops, poses_before, poses_after, outcome);
  }

  if (!batch_edges.empty()) {
    const auto before = optimizer.getOptimizedPoses();

    const auto trial = computeLoopGraphTrialDiagnostics(config, optimizer, batch_edges);
    if (trial.success && loopGraphTrialRejectReason(trial).empty() &&
        optimizer.optimizeLoopEdges(batch_edges)) {
      result.place_candidate_count += batch_loops.size();
      recordLoopOptimization(config, batch_edges, batch_loops, before,
                             optimizer.getOptimizedPoses(), outcome);
    } else {
      result.accepted_loops = std::move(batch_loops);
      rebaseLoopOutcome(config, optimizer, manager, outcome);
    }
    keyframes.updateOptimizedPoses(optimizer.getOptimizedPoses());
  }
  return outcome;
}

}
