#pragma once
#include <chrono>
#include <deque>
#include <future>
#include "loop_closure_processing.h"

namespace n3mapping {
using LoopClock = std::chrono::steady_clock;

struct AsyncLoopJob {
  AsyncLoopJob(const Config& config, const KeyframeManager& source,
               std::unique_ptr<GraphOptimizer> graph, DetectedLoopQuery query);
  void run();
  void rebase();

  Config config;
  KeyframeManager keyframes;
  PointCloudMatcher matcher;
  LoopClosureManager manager;
  std::unique_ptr<GraphOptimizer> graph;
  DetectedLoopQuery query;
  KeyframeMapRevision revision;
  LoopClosureOutcome outcome;
  double worker_wall_ms = 0.0;
  std::size_t graph_rebase_count = 0;
};

struct AsyncLoopState {
  std::uint64_t generation = 0;
  std::deque<DetectedLoopQuery> pending;
  std::unique_ptr<AsyncLoopJob> job;
  // Declared last: destruction joins the isolated worker before releasing its
  // job/snapshots. No worker needs the live owner's mutex to finish.
  std::future<void> future;
};

} // namespace n3mapping
