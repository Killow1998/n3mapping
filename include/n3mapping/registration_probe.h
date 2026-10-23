// Optional offline registration evidence; ordinary backend consumers do not need it.
#pragma once
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/visibility_consistency.h"
#include <cstdint>
#include <string>
#include <vector>
namespace n3mapping {
// Offline-only registration evidence. The oracle pose is never used by the
// runtime relocalization path; this result exists to locate which stage fails.
struct RegistrationSeedProbeAttempt {
  std::string seed_kind;
  int64_t seed_keyframe_id = -1;
  double yaw_offset_rad = 0.0;
  Eigen::Isometry3d initial_pose = Eigen::Isometry3d::Identity();
  MatchResult match;
  bool fitness_pass = false;
  bool inlier_pass = false;
  double derived_confidence = 0.0;
  bool confidence_pass = false;
  bool production_quality_pass = false;
  VisibilityConsistencyResult initial_visibility;
  VisibilityConsistencyResult refined_visibility;
  bool production_kept_initial_pose = false;
  Eigen::Isometry3d production_pose = Eigen::Isometry3d::Identity();
};

struct RegistrationSeedProbeResult {
  bool valid = false;
  std::string error;
  int64_t oracle_nearest_keyframe_id = -1;
  int64_t descriptor_candidate_keyframe_id = -1;
  std::vector<RegistrationSeedProbeAttempt> attempts;
};


}  // namespace n3mapping
