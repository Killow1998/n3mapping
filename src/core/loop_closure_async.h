#pragma once
#include <chrono>
#include <condition_variable>
#include <memory>
#include <vector>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include "loop_closure_processing.h"
#include "../background_worker.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/localization_atlas.h"

namespace n3mapping {
using LoopClock = std::chrono::steady_clock;

struct AsyncLoopJob {
  AsyncLoopJob(const Config& config, const KeyframeManager& source,
               std::unique_ptr<GraphOptimizer> graph, std::vector<DetectedLoopQuery> queries);
  void run();
  void rebase();
  void optimizeGeometry();
  bool geometryOnly() const { return queries.front().geometry_only; }

  Config config;
  KeyframeManager keyframes;
  PointCloudMatcher matcher;
  LoopClosureManager manager;
  std::unique_ptr<GraphOptimizer> graph;
  std::unique_ptr<GraphOptimizer> base_graph;
  std::unique_ptr<LocalizationAtlas> atlas;
  std::vector<DetectedLoopQuery> queries;
  KeyframeMapRevision revision;
  LoopClosureOutcome outcome;
  double worker_wall_ms = 0.0;
  double cpu_upper_ms = 0.0;
  LoopClock::time_point started;
  std::size_t graph_rebase_count = 0;
};

struct AsyncLoopState {
  std::uint64_t generation = 0;
  std::deque<DetectedLoopQuery> pending;
  LoopClock::time_point next_geometry_start;
  std::unique_ptr<AsyncLoopJob> job;
  std::future<void> future;
  // Declared last: join before releasing the job/snapshots. The worker never
  // needs the live owner's mutex to finish, including during shutdown.
  BackgroundWorker worker;
};

} // namespace n3mapping
