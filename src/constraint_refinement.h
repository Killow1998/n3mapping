// Fine registration of admitted loop measurements. The caller owns broad
// geometric admission and the loop's uncertainty model.
#pragma once

#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_quality.h"

namespace n3mapping {
inline MatchResult refineConstraint(
    const Config& config, PointCloudMatcher& matcher,
    const PointCloudMatcher::PreparedTarget& target,
    const PointCloudMatcher::PreparedSource& source,
    const Eigen::Isometry3d& initial) {
  auto settings = matcher.getSettings();
  settings.type = small_gicp::RegistrationSetting::GICP;
  settings.downsampling_resolution = config.icp_refine_downsampling_resolution;
  settings.max_correspondence_distance = config.icp_refine_max_correspondence_distance;
  settings.max_iterations = config.icp_refine_max_iterations;
  auto result = matcher.alignPrepared(target, source, initial, settings);
  const Eigen::Isometry3d delta = initial.inverse() * result.T_target_source;
  result.success =
      hasValidRegistrationResult(result) && result.converged && result.num_inliers > 0 &&
      result.fitness_score < config.loop_fitness_threshold &&
      result.inlier_ratio >= config.loop_min_inlier_ratio &&
      delta.translation().norm() <= config.icp_refine_max_correspondence_distance &&
      Eigen::AngleAxisd(delta.rotation()).angle() <= config.icp_refine_delta_rotation_gate;
  return result;
}
}  // namespace n3mapping
