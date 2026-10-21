#include "loop_closure_async.h"
#include "loop_closure_diagnostics.h"
#include <glog/logging.h>

#include <iostream>
#include <stdexcept>
#include <utility>
#include <ctime>

namespace n3mapping {
namespace {
double elapsedMs(LoopClock::time_point start) {
  return std::chrono::duration<double, std::milli>(LoopClock::now() - start).count();
}
} // namespace

AsyncLoopJob::AsyncLoopJob(const Config& config, const KeyframeManager& source,
    std::unique_ptr<GraphOptimizer> graph, DetectedLoopQuery query)
    : config(config), keyframes(config), matcher(config), manager(config),
      graph(std::move(graph)), query(std::move(query)), revision(source.revision()) {
  base_graph = this->graph->cloneCommitted();
  std::vector<Keyframe::Ptr> snapshot;
  const auto originals = source.getAllKeyframes();
  for (const auto& keyframe : originals)
    if (keyframe) snapshot.push_back(std::make_shared<Keyframe>(*keyframe));
  // Poses/metadata are copied. Point clouds are immutable after insertion and
  // can be retained across map replacement without copying their payload.
  if (!keyframes.loadKeyframes(snapshot, source.getSessions()))
    throw std::runtime_error("Failed to snapshot loop keyframes");
  // Loading intentionally marks a disk map as old-session geometry. This is
  // an in-memory snapshot instead, so retain each keyframe's original role.
  for (const auto& keyframe : originals)
    if (keyframe)
      keyframes.getKeyframe(keyframe->id)->is_from_loaded_map =
          keyframe->is_from_loaded_map;
}

void AsyncLoopJob::run() {
  const auto start = LoopClock::now();
  started = start;
  const auto cpu_start = std::clock();
  if (!query.geometry_only)
    outcome = processLoopCandidates(config, keyframes, matcher, *graph, manager, {query});
  optimizeGeometry();
  worker_wall_ms += elapsedMs(start);
  cpu_upper_ms += 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
}

void AsyncLoopJob::rebase() {
  const auto start = LoopClock::now();
  const auto cpu_start = std::clock();
  if (!query.geometry_only) rebaseLoopOutcome(config, *graph, manager, outcome);
  optimizeGeometry();
  worker_wall_ms += elapsedMs(start);
  cpu_upper_ms += 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
  ++graph_rebase_count;
}

void AsyncLoopJob::optimizeGeometry() {
  if (!config.loop_geometry_enable) return;
  keyframes.updateOptimizedPoses(graph->getOptimizedPoses());
  auto& stats = outcome.summary.geometry;
  stats = {};
  const auto planes = buildSharedPlanes(keyframes.getAllKeyframes(), stats);
  if (planes.empty()) return;
  if (!graph->replaceSharedPlanes(planes, true, stats))
    throw std::runtime_error("Joint geometry rejected; live graph unchanged");
  outcome.summary.optimized = true;
  keyframes.updateOptimizedPoses(graph->getOptimizedPoses());
  const auto residual = loop_diagnostics::meanLoopResidual(
      manager.buildLoopEdges(outcome.summary.accepted_loops, LoopEdgeDirection::MatchToQuery),
      graph->getOptimizedPoses());
  outcome.summary.loop_residual_translation_after = residual.first;
  outcome.summary.loop_residual_rotation_after = residual.second;
  std::vector<Keyframe::Ptr> loaded;
  for (const auto& frame : keyframes.getAllKeyframes())
    if (frame->is_from_loaded_map) loaded.push_back(frame);
  if (!loaded.empty()) {
    const auto start = LoopClock::now();
    atlas = std::make_unique<LocalizationAtlas>(config, matcher);
    if (!atlas->prepare(loaded))
      throw std::runtime_error("Optimized reference preparation failed");
    outcome.summary.atlas_prepare_ms = elapsedMs(start);
  }
}

CoreLoopClosureResult N3MappingCore::processPendingLoopClosuresAsync() {
  return advanceAsyncLoopClosures(false);
}

CoreLoopClosureResult N3MappingCore::advanceAsyncLoopClosures(bool wait) {
  if (!async_loops_) async_loops_ = std::make_unique<AsyncLoopState>();
  auto& state = *async_loops_;
  CoreLoopClosureResult result;
  const auto revision = session_->keyframeManager().revision();
  if (state.generation != revision.generation) {
    state.pending.clear();
    state.next_geometry_start = {};
    state.generation = revision.generation;
  }
  for (auto& query : detectPendingLoopCandidates(result)) {
    // Strict extension constraints are already in the live graph. One latest
    // geometry request covers all of them; no verified measurement is dropped.
    if (query.geometry_only) {
      for (const auto& pending : state.pending)
        if (pending.geometry_only) query.enqueued = std::min(query.enqueued, pending.enqueued);
      state.pending.erase(std::remove_if(state.pending.begin(), state.pending.end(),
          [](const auto& pending) { return pending.geometry_only; }), state.pending.end());
    }
    state.pending.push_back(std::move(query));
  }

  if (state.job && (wait ||
      state.future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
    bool completed = false;
    try {
      state.future.get();
      completed = true;
      if (state.job->query.geometry_only &&
          state.job->revision.generation == revision.generation &&
          config_.loop_geometry_cpu_budget > 0) {
        // Process CPU includes the concurrent owner and is a conservative
        // upper bound for this job, including its OpenMP feature workers.
        // All incoming constraints remain in the graph; only requests coalesce.
        const auto interval = std::chrono::duration<double, std::milli>(
            state.job->cpu_upper_ms / config_.loop_geometry_cpu_budget);
        state.next_geometry_start = state.job->started +
            std::chrono::duration_cast<LoopClock::duration>(interval);
      }
    } catch (const std::exception& error) {
      std::cerr << "Isolated loop processing failed: " << error.what() << '\n';
    }
    double catchup_ms = 0;
    if (completed && state.job->outcome.summary.optimized &&
        state.job->revision.generation == revision.generation &&
        state.job->revision != revision) {
      const auto catchup_start = LoopClock::now();
      if (state.job->graph->appendCommittedTail(session_->graphOptimizer(), *state.job->base_graph))
        state.job->revision = revision;
      catchup_ms = elapsedMs(catchup_start);
    }
    if (!completed || state.job->revision.generation != revision.generation) {
      result.stale_job_count = 1;
      result.worker_wall_ms = state.job->worker_wall_ms;
      result.graph_rebase_count = state.job->graph_rebase_count;
      state.job.reset();
    } else if (state.job->outcome.summary.optimized &&
               state.job->revision != revision) {
      // Snapshot the entire current graph, including newly appended nodes,
      // odometry, floor observations and session constraints. Only graph
      // admission/solving repeats; expensive registration evidence is retained.
      const auto start = LoopClock::now();
      auto previous = std::move(state.job);
      state.job = std::make_unique<AsyncLoopJob>(config_, session_->keyframeManager(),
          session_->graphOptimizer().cloneCommitted(), previous->query);
      state.job->outcome = std::move(previous->outcome);
      state.job->worker_wall_ms = previous->worker_wall_ms;
      state.job->cpu_upper_ms = previous->cpu_upper_ms;
      state.job->started = previous->started;
      state.job->graph_rebase_count = previous->graph_rebase_count;
      result.snapshot_ms += elapsedMs(start) + catchup_ms;
      auto* job = state.job.get();
      state.future = std::async(std::launch::async, [job] { job->rebase(); });
    } else {
      const auto detected = result.detected_candidate_count;
      const auto queued = result.queued_keyframe_count;
      const auto start = LoopClock::now();
      auto& job = *state.job;
      Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
      if (job.outcome.summary.optimized) {
        const auto before = session_->graphOptimizer().getOptimizedPoses();
        const auto after = job.graph->getOptimizedPoses();
        const int64_t anchor = session_->worldLocalizing().getLastMatchedKeyframeId();
        if (before.count(anchor) && after.count(anchor))
          correction = after.at(anchor) * before.at(anchor).inverse();
        auto& summary = job.outcome.summary;
        summary.pose_update_count = 0;
        summary.mean_pose_update_translation = summary.mean_pose_update_rotation = 0;
        summary.max_pose_update_translation = summary.max_pose_update_rotation = 0;
        loop_diagnostics::accumulatePoseUpdateStats(before, after, &summary);
        session_->graphOptimizer().swapWith(*job.graph);
      }
      publishLoopOutcome(job.outcome);
      if (job.atlas)
        session_->worldLocalizing().installOptimizedReference(*job.atlas, correction);
      result = std::move(job.outcome.summary);
      result.detected_candidate_count = detected;
      result.queued_keyframe_count = queued;
      result.commit_ms = elapsedMs(start) + catchup_ms;
      result.worker_wall_ms = job.worker_wall_ms;
      result.graph_rebase_count = job.graph_rebase_count;
      if (result.geometry.optimized)
        LOG(INFO) << "[JointGeometry] query=" << job.query.id << " planes=" << result.geometry.planes
                  << " observations=" << result.geometry.observations
                  << " association_ms=" << result.geometry.association_ms
                  << " solve_ms=" << result.geometry.solve_ms
                  << " initial_cost=" << result.geometry.initial_cost
                  << " final_cost=" << result.geometry.final_cost
                  << " geometry_max_t=" << result.geometry.max_translation_update_m
                  << " geometry_max_r=" << result.geometry.max_rotation_update_rad
                  << " total_max_t=" << result.max_pose_update_translation
                  << " atlas_ms=" << result.atlas_prepare_ms
                  << " worker_ms=" << result.worker_wall_ms
                  << " cpu_upper_ms=" << job.cpu_upper_ms
                  << " commit_ms=" << result.commit_ms
                  << " rebases=" << result.graph_rebase_count;
      state.job.reset();
    }
  }
  if (!state.job && !state.pending.empty() &&
      (wait || !state.pending.front().geometry_only ||
       LoopClock::now() >= state.next_geometry_start)) {
    const auto start = LoopClock::now();
    auto query = std::move(state.pending.front());
    state.pending.pop_front();
    state.job = std::make_unique<AsyncLoopJob>(
        config_, session_->keyframeManager(),
        session_->graphOptimizer().cloneCommitted(), std::move(query));
    result.snapshot_ms += elapsedMs(start);
    auto* job = state.job.get();
    state.future = std::async(std::launch::async, [job] { job->run(); });
  }
  result.pending_query_count = state.pending.size() + (state.job ? 1 : 0);
  result.async_pending = result.pending_query_count != 0;
  if (state.job)
    result.oldest_pending_age_ms = elapsedMs(state.job->query.enqueued);
  else if (!state.pending.empty())
    result.oldest_pending_age_ms = elapsedMs(state.pending.front().enqueued);
  return result;
}

} // namespace n3mapping
