// Geometry and graph work shared by synchronous replay and the isolated worker.
#pragma once
#include <chrono>
#include "n3mapping/core/n3mapping_core.h"
namespace n3mapping {
struct DetectedLoopQuery {
  int64_t id = -1;
  std::vector<LoopCandidate> candidates;
  std::chrono::steady_clock::time_point enqueued =
      std::chrono::steady_clock::now();
  bool geometry_only = false;
};
struct LoopClosureOutcome {
  CoreLoopClosureResult summary;
  std::vector<LoopDebugCandidateEvent> candidates;
  std::vector<LoopDebugOptimizationEvent> optimizations;
};
void recordLoopOptimization(
    const Config& config, const std::vector<EdgeInfo>& edges,
    const std::vector<VerifiedLoop>& loops,
    const std::map<int64_t, Eigen::Isometry3d>& before,
    const std::map<int64_t, Eigen::Isometry3d>& after,
    LoopClosureOutcome& outcome);
LoopClosureOutcome processLoopCandidates(
    const Config& config, KeyframeManager& keyframes, PointCloudMatcher& matcher,
    GraphOptimizer& optimizer, LoopClosureManager& manager,
    const std::vector<DetectedLoopQuery>& queries);
void rebaseLoopOutcome(const Config& config, GraphOptimizer& optimizer,
                       LoopClosureManager& manager, LoopClosureOutcome& outcome);
} // namespace n3mapping
