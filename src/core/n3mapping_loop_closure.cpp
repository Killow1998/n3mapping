#include "n3mapping/core/n3mapping_core.h"
#include "loop_closure_processing.h"
#include "loop_closure_async.h"
#include <algorithm>

namespace n3mapping {
void N3MappingCore::appendLoopDebugCandidate(
    const LoopDebugCandidateEvent &event) const {
  if (!config_.loop_debug_enable) {
    return;
  }
  std::lock_guard<std::mutex> lock(loop_debug_mutex_);
  LoopDebugLogger::appendCandidate(LoopDebugLogger::resolvePath(config_),
                                   event);
}

void N3MappingCore::appendLoopDebugOptimization(
    const LoopDebugOptimizationEvent &event) const {
  if (!config_.loop_debug_enable) {
    return;
  }
  std::lock_guard<std::mutex> lock(loop_debug_mutex_);
  LoopDebugLogger::appendOptimizationSummary(
      LoopDebugLogger::resolvePath(config_), event);
}

std::vector<DetectedLoopQuery> N3MappingCore::detectPendingLoopCandidates(
    CoreLoopClosureResult& result) {
  std::vector<int64_t> keyframes_to_check;
  { std::lock_guard<std::mutex> lock(loop_queue_mutex_);
    keyframes_to_check.swap(loop_detection_queue_); }
  result.queued_keyframe_count = keyframes_to_check.size();
  std::vector<DetectedLoopQuery> queries;
  if (map_loaded_ && config_.loop_geometry_enable && !keyframes_to_check.empty()) {
    DetectedLoopQuery query;
    query.id = keyframes_to_check.back();
    query.geometry_only = true;
    queries.push_back(std::move(query));
    return queries;
  }
  for (int64_t query_id : keyframes_to_check) {
    if (query_id - last_loop_check_id_ < config_.loop_kf_gap) {
      continue;
    }

    auto query_kf = session_->keyframeManager().getKeyframe(query_id);
    if (!query_kf) {
      continue;
    }

    std::map<int64_t, Keyframe::Ptr> keyframe_map;
    for (const auto &keyframe : session_->keyframeManager().getAllKeyframes()) {
      if (keyframe) {
        keyframe_map[keyframe->id] = keyframe;
      }
    }
    const auto within_work_bound = [&](int64_t match_id) {
      const auto match = keyframe_map.find(match_id);
      if (match == keyframe_map.end()) return false;
      const auto predicted = match->second->pose_optimized.inverse() * query_kf->pose_optimized;
      return !(predicted.translation().norm() > config_.loop_max_range);
    };
    std::vector<LoopCandidate> candidates =
        session_->loopDetector().detectLoopCandidates(query_id, keyframe_map, {},
                                                      within_work_bound);
    if (config_.loop_spatial_candidates_enable) {
      auto spatial_candidates =
          session_->loopDetector().detectSpatialCandidates(query_id,
                                                           keyframe_map, {}, true);
      for (const auto &spatial : spatial_candidates) {
        auto duplicate =
            std::find_if(candidates.begin(), candidates.end(),
                         [&](const LoopCandidate &existing) {
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
    result.detected_candidate_count += candidates.size();
    if (candidates.empty()) {
      continue;
    }
    last_loop_check_id_ = query_id;
    queries.push_back({query_id, std::move(candidates)});
  }

  return queries;
}

void N3MappingCore::publishLoopOutcome(const LoopClosureOutcome& outcome) {
  loop_count_ += outcome.summary.edge_count;
  if (outcome.summary.optimized) refreshOptimizedPoses("loop_commit");
  for (const auto& event : outcome.candidates) appendLoopDebugCandidate(event);
  for (const auto& event : outcome.optimizations) appendLoopDebugOptimization(event);
}

CoreLoopClosureResult N3MappingCore::processPendingLoopClosures() {
  if (config_.loop_geometry_enable && !async_loops_)
    async_loops_ = std::make_unique<AsyncLoopState>();
  if (async_loops_) {
    CoreLoopClosureResult total;
    CoreLoopClosureResult completed;
    do {
      completed = advanceAsyncLoopClosures(true);
      total.optimized |= completed.optimized;
      total.queued_keyframe_count += completed.queued_keyframe_count;
      total.detected_candidate_count += completed.detected_candidate_count;
      total.place_candidate_count += completed.place_candidate_count;
      total.edge_count += completed.edge_count;
      total.graph_edge_count += completed.graph_edge_count;
      total.worker_wall_ms += completed.worker_wall_ms;
      total.snapshot_ms += completed.snapshot_ms;
      total.commit_ms += completed.commit_ms;
      total.geometry.association_ms += completed.geometry.association_ms;
      total.geometry.solve_ms += completed.geometry.solve_ms;
      total.geometry.optimized |= completed.geometry.optimized;
      total.atlas_prepare_ms += completed.atlas_prepare_ms;
      if (completed.geometry.optimized) {
        total.geometry.planes = completed.geometry.planes;
        total.geometry.observations = completed.geometry.observations;
      }
      total.graph_rebase_count += completed.graph_rebase_count;
      total.stale_job_count += completed.stale_job_count;
      const auto pose_count = total.pose_update_count + completed.pose_update_count;
      if (pose_count > 0) {
        total.mean_pose_update_translation =
            (total.mean_pose_update_translation * total.pose_update_count +
             completed.mean_pose_update_translation * completed.pose_update_count) / pose_count;
        total.mean_pose_update_rotation =
            (total.mean_pose_update_rotation * total.pose_update_count +
             completed.mean_pose_update_rotation * completed.pose_update_count) / pose_count;
      }
      total.pose_update_count = pose_count;
      total.max_pose_update_translation = std::max(
          total.max_pose_update_translation, completed.max_pose_update_translation);
      total.max_pose_update_rotation = std::max(
          total.max_pose_update_rotation, completed.max_pose_update_rotation);
      if (completed.optimized) {
        total.loop_residual_translation_before = completed.loop_residual_translation_before;
        total.loop_residual_translation_after = completed.loop_residual_translation_after;
        total.loop_residual_rotation_before = completed.loop_residual_rotation_before;
        total.loop_residual_rotation_after = completed.loop_residual_rotation_after;
      }
      total.accepted_loops.insert(total.accepted_loops.end(),
          completed.accepted_loops.begin(), completed.accepted_loops.end());
    } while (completed.async_pending);
    return total;
  }
  CoreLoopClosureResult detection;
  const auto queries = detectPendingLoopCandidates(detection);
  auto outcome = processLoopCandidates(config_, session_->keyframeManager(),
      session_->pointCloudMatcher(), session_->graphOptimizer(),
      session_->loopClosureManager(), queries);
  publishLoopOutcome(outcome);
  outcome.summary.queued_keyframe_count = detection.queued_keyframe_count;
  outcome.summary.detected_candidate_count = detection.detected_candidate_count;
  return outcome.summary;
}
} // namespace n3mapping
