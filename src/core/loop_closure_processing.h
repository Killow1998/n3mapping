#pragma once
#include <chrono>
#include <cmath>
#include "n3mapping/keyframe.h"
#include "n3mapping/visibility_consistency.h"
#include "n3mapping/core/loop_closure_result.h"
#include "n3mapping/loop_closure_manager.h"
#include "n3mapping/loop_debug_logger.h"
namespace n3mapping {
class GraphOptimizer;
class KeyframeManager;
class PointCloudMatcher;
inline constexpr double kMaxLoopVerticalSeparationM = 2.0;

inline bool eligibleOldMapLoopCandidate(const Keyframe& match, const Keyframe& query,
                                        double horizontal_radius) {
  if (!match.is_from_loaded_map) return false;
  const Eigen::Vector3d delta = match.getPosition() - query.getPosition();
  return std::abs(delta.z()) <= kMaxLoopVerticalSeparationM &&
      delta.head<2>().squaredNorm() <= horizontal_radius * horizontal_radius;
}
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
    const std::vector<DetectedLoopQuery>& queries,
    const LoadedMapVisibilityEvaluator& visibility = {});
void rebaseLoopOutcome(const Config& config, GraphOptimizer& optimizer,
                       LoopClosureManager& manager, LoopClosureOutcome& outcome);
}
