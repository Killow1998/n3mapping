#include "loop_closure_async.h"

#include <iostream>
#include <stdexcept>
#include <utility>

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
  outcome = processLoopCandidates(config, keyframes, matcher, *graph, manager, {query});
  worker_wall_ms += elapsedMs(start);
}

void AsyncLoopJob::rebase() {
  const auto start = LoopClock::now();
  rebaseLoopOutcome(config, *graph, manager, outcome);
  worker_wall_ms += elapsedMs(start);
  ++graph_rebase_count;
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
    state.generation = revision.generation;
  }
  for (auto& query : detectPendingLoopCandidates(result))
    state.pending.push_back(std::move(query));

  if (state.job && (wait ||
      state.future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
    bool completed = false;
    try {
      state.future.get();
      completed = true;
    } catch (const std::exception& error) {
      std::cerr << "Isolated loop processing failed: " << error.what() << '\n';
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
      state.job->graph = session_->graphOptimizer().cloneCommitted();
      state.job->revision = revision;
      result.snapshot_ms += elapsedMs(start);
      auto* job = state.job.get();
      state.future = std::async(std::launch::async, [job] { job->rebase(); });
    } else {
      const auto detected = result.detected_candidate_count;
      const auto queued = result.queued_keyframe_count;
      const auto start = LoopClock::now();
      auto& job = *state.job;
      if (job.outcome.summary.optimized)
        session_->graphOptimizer().swapWith(*job.graph);
      publishLoopOutcome(job.outcome);
      result = std::move(job.outcome.summary);
      result.detected_candidate_count = detected;
      result.queued_keyframe_count = queued;
      result.commit_ms = elapsedMs(start);
      result.worker_wall_ms = job.worker_wall_ms;
      result.graph_rebase_count = job.graph_rebase_count;
      state.job.reset();
    }
  }
  if (!state.job && !state.pending.empty()) {
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
