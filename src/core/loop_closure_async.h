#pragma once
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include "loop_closure_processing.h"

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

// One isolated job at a time, on a reusable thread. Keeping its OpenMP team
// and allocation history avoids rebuilding them for every loop snapshot.
class AsyncLoopWorker {
public:
  ~AsyncLoopWorker() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    ready_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

  template <class Function>
  std::future<void> submit(Function&& function) {
    std::packaged_task<void()> next(std::forward<Function>(function));
    auto result = next.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_ || task_.valid()) throw std::logic_error("Loop worker already queued");
      if (!thread_.joinable()) thread_ = std::thread([this] { run(); });
      task_ = std::move(next);
    }
    ready_.notify_one();
    return result;
  }

private:
  void run() {
    for (;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || task_.valid(); });
        if (!task_.valid()) return;
        task = std::move(task_);
      }
      task();  // Exceptions are delivered by the existing future.get() path.
    }
  }
  std::mutex mutex_;
  std::condition_variable ready_;
  bool stopping_ = false;
  std::packaged_task<void()> task_;
  std::thread thread_;
};

struct AsyncLoopState {
  std::uint64_t generation = 0;
  std::deque<DetectedLoopQuery> pending;
  LoopClock::time_point next_geometry_start;
  std::unique_ptr<AsyncLoopJob> job;
  std::future<void> future;
  // Declared last: join before releasing the job/snapshots. The worker never
  // needs the live owner's mutex to finish, including during shutdown.
  AsyncLoopWorker worker;
};

} // namespace n3mapping
