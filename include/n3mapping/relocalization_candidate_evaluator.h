// Geometric evaluation of one relocalization place candidate.
#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_target_provider.h"
#include "n3mapping/visibility_consistency.h"

namespace n3mapping {

struct RegistrationEvidence {
  MatchResult match;
  bool production_quality = false;
  Eigen::Isometry3d initial_pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d selected_pose = Eigen::Isometry3d::Identity();
  bool selected_refined = false;

  MatchResult selectedMatch() const {
    MatchResult selected = match;
    selected.T_target_source = selected_pose;
    return selected;
  }
};

struct RelocalizationCandidateEvaluation {
  RegistrationEvidence registration;
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

  std::vector<RelocalizationCandidateEvaluation>
  evaluate(const PointCloudT::Ptr &query_cloud,
           const PointCloudMatcher::PreparedSource &prepared_query,
           const LoopCandidate &candidate, RelocTargetRequest target_request);

private:
  VisibilityConsistencyResult
  evaluateVisibility(const PointCloudT::Ptr &target_cloud,
                     const PointCloudT::Ptr &query_cloud,
                     const Eigen::Isometry3d &T_map_lidar) const;

  Config config_;
  KeyframeManager &keyframe_manager_;
  PointCloudMatcher &matcher_;
  RelocTargetProvider &target_provider_;
};

} // namespace n3mapping
