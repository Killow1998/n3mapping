#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include "n3mapping/keyframe.h"
#include "n3mapping/visibility_consistency.h"
#include "n3mapping/core/loop_closure_result.h"
#include "n3mapping/loop_closure_manager.h"
#include "n3mapping/loop_debug_logger.h"
namespace n3mapping {
class GraphOptimizer;
class KeyframeManager;
class PointCloudMatcher;
inline double maxVerticalDrift(double path_length) {
  return 0.75 + 0.02 * std::max(0.0, path_length);
}

inline double maxHorizontalDrift(double path_length, double cap) {
  return std::min(cap, 1.5 + 0.05 * std::max(0.0, path_length));
}

struct LoopOdometryPath {
  double length;
  double untracked;
};
using LoopOdometryPaths = std::unordered_map<int64_t, LoopOdometryPath>;
using LoopAnchorOffsets = std::unordered_map<int64_t, double>;

LoopOdometryPaths measureLoopOdometryPaths(
    const std::vector<Keyframe::Ptr>& keyframes,
    const LoopAnchorOffsets& anchor_offsets);

struct LoopDriftBudget {
  LoopOdometryPaths paths;
  std::unordered_map<MapSessionId, double> accepted_anchors;

  double untracked(const Keyframe& query) const {
    const auto& path = paths.at(query.id);
    const auto anchor = accepted_anchors.find(query.session_id);
    return anchor == accepted_anchors.end() ? path.untracked
        : std::min(path.untracked, path.length - anchor->second);
  }
  void anchor(const Keyframe& query) {
    accepted_anchors[query.session_id] = paths.at(query.id).length;
  }
};

inline bool withinSameSessionDriftEnvelope(const Keyframe& match, const Keyframe& query,
                                           const LoopOdometryPaths& paths) {
  if (match.session_id != query.session_id) return false;
  const double path_length = std::abs(paths.at(query.id).length - paths.at(match.id).length);
  return std::abs(match.getPosition().z() - query.getPosition().z()) <= maxVerticalDrift(path_length);
}

inline bool eligibleOldMapLoopCandidate(const Keyframe& match, const Keyframe& query,
                                        double untracked_distance, double horizontal_radius) {
  if (!match.is_from_loaded_map) return false;
  const Eigen::Vector3d delta = match.getPosition() - query.getPosition();
  return std::abs(delta.z()) <= maxVerticalDrift(untracked_distance) &&
      delta.head<2>().norm() <= maxHorizontalDrift(untracked_distance, horizontal_radius);
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
    const LoadedMapVisibilityEvaluator& visibility = {},
    const LoopAnchorOffsets& anchor_offsets = {});
void rebaseLoopOutcome(const Config& config, GraphOptimizer& optimizer,
                       const KeyframeManager& keyframes, LoopClosureManager& manager,
                       LoopClosureOutcome& outcome, const LoopAnchorOffsets& anchor_offsets);
}
