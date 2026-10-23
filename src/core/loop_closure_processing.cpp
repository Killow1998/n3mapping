#include "loop_closure_processing.h"
#include "loop_closure_diagnostics.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <glog/logging.h>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/loop_consensus_verifier.h"
#include "n3mapping/loop_graph_trial_diagnostics.h"
#include "n3mapping/loop_verification_pipeline.h"

namespace n3mapping {
using namespace loop_diagnostics;

LoopClosureOutcome processLoopCandidates(
    const Config& config, KeyframeManager& keyframes, PointCloudMatcher& matcher,
    GraphOptimizer& optimizer, LoopClosureManager& manager,
    const std::vector<DetectedLoopQuery>& queries) {
  LoopClosureOutcome outcome;
  auto& result = outcome.summary;
  LoopVerificationPipeline verification_pipeline(config, keyframes, matcher, optimizer, manager);
  std::vector<EdgeInfo> batch_edges;
  std::vector<VerifiedLoop> batch_loops;
  for (const auto& query : queries) {
    const auto query_kf = keyframes.getKeyframe(query.id);
    if (!query_kf) continue;
    const auto& candidates = query.candidates;
    std::map<int64_t, Keyframe::Ptr> keyframe_map;
    for (const auto& keyframe : keyframes.getAllKeyframes())
      if (keyframe) keyframe_map.emplace(keyframe->id, keyframe);
    std::vector<VerifiedLoop> verified_loops;
    verified_loops.reserve(candidates.size());
    std::vector<LoopDebugCandidateEvent> debug_events;
    std::map<std::pair<int64_t, int64_t>, EdgeInfo> edge_by_pair;
    std::map<std::pair<int64_t, int64_t>, LoopGraphTrialDiagnostics>
        graph_trial_by_pair;
    std::map<std::pair<int64_t, int64_t>, LoopGraphTrialDiagnostics>
        consensus_estimator_trial_by_pair;
    std::map<std::pair<int64_t, int64_t>, LoopRefereeDebugDecision>
        referee_by_pair;
    std::map<std::pair<int64_t, int64_t>, LoopConsensusResult>
        consensus_by_pair;
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
            if (const auto edge = edge_by_pair.find(key); edge != edge_by_pair.end())
              event.loop_information = edge->second.information;
            auto trial_it = graph_trial_by_pair.find(key);
            if (trial_it != graph_trial_by_pair.end()) {
              assignGraphTrialDiagnostics(&event, trial_it->second);
            }
            auto consensus_trial_it =
                consensus_estimator_trial_by_pair.find(key);
            if (consensus_trial_it != consensus_estimator_trial_by_pair.end()) {
              assignConsensusEstimatorTrialDiagnostics(
                  &event, consensus_trial_it->second);
            }
            auto referee_it = referee_by_pair.find(key);
            if (referee_it != referee_by_pair.end()) {
              event.loop_referee_recommendation =
                  referee_it->second.recommendation;
              event.loop_referee_reason = referee_it->second.reason;
              event.loop_referee_risk_flags = referee_it->second.risk_flags;
            }
            auto consensus_it = consensus_by_pair.find(key);
            if (consensus_it != consensus_by_pair.end()) {
              assignConsensusDiagnostics(&event, consensus_it->second);
            }
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
            outcome.candidates.push_back(event);
          }
        };

    auto make_rejected_event = [&](const LoopCandidate &candidate,
                                   const std::string &reject_reason) {
      if (!loop_debug_enabled) {
        return;
      }
      LoopDebugCandidateEvent event;
      event.processing_time = processingTimeSeconds();
      event.query_timestamp = query_kf->timestamp;
      event.candidate = candidate;
      event.gate_result = "rejected";
      event.reject_reason = reject_reason;
      debug_events.push_back(event);
    };

    for (const auto &candidate : candidates) {
      auto pipeline_result = verification_pipeline.evaluate(candidate);
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
      if (!pipeline_result.registration_attempted) {
        make_rejected_event(candidate, pipeline_result.reject_reason);
        continue;
      }
      const auto &source = pipeline_result.source_registration_cloud;
      const auto &target = pipeline_result.target_registration_cloud;
      ZDistributionStats source_z_before;
      ZDistributionStats target_z;
      if (loop_debug_enabled) {
        source_z_before = computeZStats(source);
        target_z = computeZStats(target);
      }

      LoopVerification verification = pipeline_result.verification;
      MatchResult match_result = verification.match_result;
      ZDistributionStats source_z_after;
      if (loop_debug_enabled) {
        source_z_after = computeTransformedZStats(
            source, verification.T_icp_correction_match);
      }

      VerifiedLoop loop = pipeline_result.loop;
      loop.source_z_span = source_z_before.span();
      loop.target_z_span = target_z.span();
      loop.z_overlap_ratio_before = zOverlapRatio(source_z_before, target_z);
      loop.z_overlap_ratio_after = zOverlapRatio(source_z_after, target_z);
      loop.source_z_robust_span = source_z_before.robustSpan();
      loop.target_z_robust_span = target_z.robustSpan();
      loop.z_robust_overlap_ratio_before =
          robustZOverlapRatio(source_z_before, target_z);
      loop.z_robust_overlap_ratio_after =
          robustZOverlapRatio(source_z_after, target_z);
      loop.source_target_z_centroid_delta_before =
          zCentroidDelta(source_z_before, target_z);
      loop.source_target_z_centroid_delta_after =
          zCentroidDelta(source_z_after, target_z);
      if (loop_debug_enabled && config.loop_debug_vertical_hypotheses_enable &&
          match_result.converged && verification.fitness_ok &&
          verification.inlier_ok && verification.geometry_ok) {
        const auto diagnostics = computeVerticalHypothesisDiagnostics(
            matcher, target, source, match_result);
        loop.vertical_hypothesis_count = diagnostics.count;
        loop.best_z_offset_m = diagnostics.best_z_offset_m;
        loop.best_z_offset_fitness = diagnostics.best_z_offset_fitness;
        loop.zero_z_fitness = diagnostics.zero_z_fitness;
        loop.fitness_gap_zero_vs_best = diagnostics.fitness_gap_zero_vs_best;
        loop.z_hypothesis_spread_m = diagnostics.z_hypothesis_spread_m;
        loop.vertical_ambiguity_score = diagnostics.vertical_ambiguity_score;
        loop.vertical_hypothesis_edge_recommendation =
            diagnostics.edge_recommendation;
      }
      const std::string reject_reason = pipeline_result.reject_reason;
      if (loop_debug_enabled) {
        LoopDebugCandidateEvent event;
        event.processing_time = processingTimeSeconds();
        event.query_timestamp = query_kf->timestamp;
        event.candidate = candidate;
        event.icp_converged = match_result.converged;
        event.icp_iterations = match_result.iterations;
        event.icp_optimizer_error = match_result.optimizer_error;
        event.icp_termination = matchTerminationName(match_result.termination);
        event.fitness_score = match_result.fitness_score;
        event.inlier_ratio = match_result.inlier_ratio;
        event.icp_translation_norm = verification.icp_translation_norm;
        event.icp_rotation_norm = verification.icp_rotation_norm;
        event.residual = verification.T_measurement_residual;
        event.T_pred_match_query = verification.T_pred_match_query;
        event.T_icp_correction_match = verification.T_icp_correction_match;
        event.T_measured_match_query = verification.T_measured_match_query;
        event.has_loop_measurement = true;
        event.loop_measurement_match_query =
            verification.T_measured_match_query;
        event.loop_information = loop.information;
        event.edge_mode =
            loop.verified ? loopEdgeModeName(loop.edge_mode) : "not_applicable";
        event.vertical_observability_score =
            loop.verified ? loop.vertical_observability_score
                          : std::numeric_limits<double>::quiet_NaN();
        event.vertical_downweighted = loop.vertical_downweighted;
        event.source_z_span = loop.source_z_span;
        event.target_z_span = loop.target_z_span;
        event.z_overlap_ratio_before = loop.z_overlap_ratio_before;
        event.z_overlap_ratio_after = loop.z_overlap_ratio_after;
        event.source_z_robust_span = loop.source_z_robust_span;
        event.target_z_robust_span = loop.target_z_robust_span;
        event.z_robust_overlap_ratio_before =
            loop.z_robust_overlap_ratio_before;
        event.z_robust_overlap_ratio_after = loop.z_robust_overlap_ratio_after;
        event.source_target_z_centroid_delta_before =
            loop.source_target_z_centroid_delta_before;
        event.source_target_z_centroid_delta_after =
            loop.source_target_z_centroid_delta_after;
        event.vertical_information_ratio = loop.vertical_information_ratio;
        event.vertical_hypothesis_count = loop.vertical_hypothesis_count;
        event.best_z_offset_m = loop.best_z_offset_m;
        event.best_z_offset_fitness = loop.best_z_offset_fitness;
        event.zero_z_fitness = loop.zero_z_fitness;
        event.fitness_gap_zero_vs_best = loop.fitness_gap_zero_vs_best;
        event.z_hypothesis_spread_m = loop.z_hypothesis_spread_m;
        event.vertical_ambiguity_score = loop.vertical_ambiguity_score;
        event.vertical_hypothesis_edge_recommendation =
            loop.vertical_hypothesis_edge_recommendation;
        event.heightmap_overlap_cell_count = loop.heightmap_overlap_cell_count;
        event.heightmap_overlap_ratio = loop.heightmap_overlap_ratio;
        event.heightmap_ground_dz_median = loop.heightmap_ground_dz_median;
        event.heightmap_ground_dz_p90 = loop.heightmap_ground_dz_p90;
        event.heightmap_ground_dz_max = loop.heightmap_ground_dz_max;
        event.heightmap_ground_support_ratio =
            loop.heightmap_ground_support_ratio;
        event.heightmap_vertical_consistency_score =
            loop.heightmap_vertical_consistency_score;
        assignSegmentDiagnostics(&event, loop);
        assignConsensusDiagnostics(&event, loop);
        event.loop_referee_recommendation = loop.loop_referee_recommendation;
        event.loop_referee_reason = loop.loop_referee_reason;
        event.loop_referee_risk_flags = loop.loop_referee_risk_flags;
        event.gate_result = loop.verified ? "accepted" : "rejected";
        event.reject_reason = reject_reason;
        debug_events.push_back(event);
      }
      verified_loops.push_back(loop);
    }

    if (verified_loops.empty()) {
      flush_debug_events({}, "not_selected");
      continue;
    }

    auto valid_loops =
        manager.filterValidLoops(verified_loops);
    // Run every individual constraint gate before same-query consensus so a
    // candidate that is already rejected cannot manufacture or break a
    // majority. The legacy single-best mode retains its historical pre-gate
    // selection order.
    auto candidate_loops =
        config.loop_keep_all_verified
            ? valid_loops
            : manager.selectBestPerQuery(valid_loops);
    if (candidate_loops.empty()) {
      flush_debug_events({}, "not_selected");
      continue;
    }

    const auto poses_before = optimizer.getOptimizedPoses();
    std::vector<VerifiedLoop> gated_loops;
    gated_loops.reserve(candidate_loops.size());

    for (const auto &gate : verification_pipeline.evaluateConstraints(
             candidate_loops, LoopEdgeDirection::MatchToQuery)) {
      const auto key = std::make_pair(gate.loop.query_id, gate.loop.match_id);
      consensus_by_pair[key] = gate.consensus;
      if (gate.has_edge) {
        edge_by_pair[key] = gate.edge;
        graph_trial_by_pair[key] = gate.graph_trial;
      }
      if (gate.has_consensus_estimator_trial) {
        consensus_estimator_trial_by_pair[key] =
            gate.consensus_estimator_trial;
      }
      referee_by_pair[key] = {gate.loop.loop_referee_recommendation,
                              gate.loop.loop_referee_reason,
                              gate.loop.loop_referee_risk_flags};
      if (!gate.accepted || !gate.has_edge) {
        graph_reject_by_pair[key] = gate.reject_reason.empty()
                                        ? "verification_pipeline_rejected"
                                        : gate.reject_reason;
        continue;
      }
      gated_loops.push_back(gate.loop);
    }

    if (gated_loops.empty()) {
      flush_debug_events({}, "verification_pipeline_rejected");
      continue;
    }

    // Preserve all measurements in the strict-majority same-query consensus
    // set. This retains redundant constraints without committing an isolated
    // corridor alias beside several agreeing measurements. Ambiguous sets
    // fall back to the historical best-per-query policy.
    std::vector<VerifiedLoop> best_loops;
    if (config.loop_keep_all_verified) {
      auto selection = manager.selectSameQueryConsensus(
          gated_loops, keyframe_map);
      best_loops = std::move(selection.selected);
      graph_reject_by_pair.insert(selection.rejected.begin(),
                                  selection.rejected.end());
    } else {
      best_loops = std::move(gated_loops);
    }

    std::vector<EdgeInfo> edges;
    std::vector<VerifiedLoop> loops_with_edges;
    edges.reserve(best_loops.size());
    loops_with_edges.reserve(best_loops.size());
    for (const auto &loop : best_loops) {
      const auto key = std::make_pair(loop.query_id, loop.match_id);
      const auto edge_it = edge_by_pair.find(key);
      if (edge_it == edge_by_pair.end()) {
        graph_reject_by_pair[key] = "same_query_consensus_missing_edge";
        continue;
      }
      edges.push_back(edge_it->second);
      loops_with_edges.push_back(loop);
    }
    best_loops.swap(loops_with_edges);
    if (edges.empty()) {
      flush_debug_events({}, "same_query_consensus_rejected");
      continue;
    }
    if (config.loop_geometry_enable && queries.size() > 1) {
      batch_edges.insert(batch_edges.end(), edges.begin(), edges.end());
      batch_loops.insert(batch_loops.end(), best_loops.begin(), best_loops.end());
      std::set<std::pair<int64_t, int64_t>> pairs;
      for (const auto& loop : best_loops) pairs.emplace(loop.query_id, loop.match_id);
      flush_debug_events(pairs, "not_selected");
      continue;
    }
    result.place_candidate_count += best_loops.size();
    const bool optimization_committed =
        manager.applyEdges(edges,
                                                  optimizer);
    if (!optimization_committed) {
      flush_debug_events({}, "optimization_failed");
      continue;
    }

    std::set<std::pair<int64_t, int64_t>> accepted_debug_pairs;
    for (const auto &loop : best_loops) {
      accepted_debug_pairs.insert({loop.query_id, loop.match_id});
    }
    flush_debug_events(accepted_debug_pairs, "not_selected");

    keyframes.updateOptimizedPoses(optimizer.getOptimizedPoses());
    const auto poses_after = optimizer.getOptimizedPoses();

    recordLoopOptimization(config, edges, best_loops, poses_before, poses_after, outcome);
  }

  if (!batch_edges.empty()) {
    const auto before = optimizer.getOptimizedPoses();
    // Individual geometry/consensus/admission gates already passed. Check the
    // combined constraints too before sharing one full graph solve. A conflict
    // falls back per query using the same measurements, without another ICP.
    const auto trial = computeLoopGraphTrialDiagnostics(
        config, before, optimizer.getEdges(), batch_edges);
    if (loopGraphTrialRejectReason(trial, true).empty() && manager.applyEdges(batch_edges, optimizer)) {
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

} // namespace n3mapping
