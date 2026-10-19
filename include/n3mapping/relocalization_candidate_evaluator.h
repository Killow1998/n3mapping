// Geometric evaluation of one relocalization place candidate.
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_target_provider.h"
#include "n3mapping/relocalization_performance.h"
#include "n3mapping/visibility_consistency.h"

namespace n3mapping {

struct RelocalizationCandidateEvaluation {
  // The selected pose and its own metrics. A fallback is a fixed-pose
  // evaluation, not a second optimizer result or a separately stored pose.
  MatchResult match;
  int64_t matched_keyframe_id = -1;
  VisibilityConsistencyResult visibility;
};

class RelocalizationCandidateEvaluator {
public:
  using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

  RelocalizationCandidateEvaluator(const Config &config,
                                   KeyframeManager &keyframe_manager,
                                   PointCloudMatcher &matcher,
                                   RelocTargetProvider &target_provider);

  // Called before parallel evaluation. Retain the current and previous seed
  // plans so a one-window ranking change does not discard a useful proposal.
  // A stored pose carries no confirmation evidence. Unconverged endpoints
  // require a new full registration before they can produce a hypothesis.
  void beginSeedPlan(const std::vector<std::pair<int64_t, double>>& seeds,
                     const KeyframeMapRevision& revision);
  void clear();
  std::size_t discardSeedsNear(const Eigen::Vector3d& center, double radius_xy);

  std::vector<RelocalizationCandidateEvaluation>
  evaluate(const PointCloudMatcher::PreparedSource &prepared_query,
           const LoopCandidate &candidate, RelocTargetRequest target_request,
           const PreparedVisibilityObservation &visibility_observation,
           const Eigen::Isometry3d &odom_pose,
           RelocalizationPerformance *performance = nullptr);

private:
  Config config_;
  KeyframeManager &keyframe_manager_;
  PointCloudMatcher &matcher_;
  RelocTargetProvider &target_provider_;
  std::mutex seed_mutex_;
  KeyframeMapRevision seed_revision_;
  std::vector<std::pair<int64_t, double>> previous_seeds_;
  struct SeedPose {
    Eigen::Isometry3d map_from_odom;
    MatchMetric metric;
    bool needs_registration = false;
  };
  std::map<std::pair<int64_t, double>, SeedPose> seed_poses_;
};

} // namespace n3mapping
