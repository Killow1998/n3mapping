#include "n3mapping/world_localizing.h"
#include "relocalization_seed_plan.h"
#include "relocalization_observation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <glog/logging.h>

namespace n3mapping {
RegistrationSeedProbeResult
WorldLocalizing::probeRegistrationSeeds(const PointCloudT::Ptr &cloud,
                                        const Eigen::Isometry3d &odom_pose,
                                        const Eigen::Isometry3d &oracle_pose) {
  RegistrationSeedProbeResult result;
  if (!cloud || cloud->empty()) {
    result.error = "empty_cloud";
    return result;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (keyframe_manager_.size() == 0) {
    result.error = "missing_keyframes";
    return result;
  }
  if (!localization_atlas_ || !localization_atlas_->loaded()) {
    result.error = "atlas_required";
    return result;
  }

  auto query_cloud = query_builder_.buildStationary(cloud, odom_pose).cloud;
  if (!query_cloud || query_cloud->empty()) {
    result.error = "empty_query_cloud";
    return result;
  }
  RelocalizationObservation observation;
  RelocalizationPerformance preparation_cost;
  observation.prepare(query_cloud, matcher_, config_, preparation_cost);
  const auto& prepared_query = observation.registration;
  const auto& visibility_observation = observation.visibility;
  const auto &target = localization_atlas_->preparedTarget();

  const auto finish_attempt = [&](RegistrationSeedProbeAttempt *attempt) {
    if (!attempt)
      return;
    const auto quality = evaluateRelocMatchQuality(attempt->match, config_);
    attempt->fitness_pass = quality.fitness_pass;
    attempt->inlier_pass = quality.inlier_pass;
    attempt->derived_confidence = quality.confidence;
    attempt->confidence_pass = quality.confidence_pass;
    attempt->production_quality_pass = quality.accepted;
    const auto visibility_target =
        buildRelocTargetCloud(attempt->seed_keyframe_id);
    if (visibility_target) {
      attempt->initial_visibility = evaluatePreparedVisibilityConsistency(
          *visibility_target, visibility_observation, attempt->initial_pose);
      attempt->refined_visibility = evaluatePreparedVisibilityConsistency(
          *visibility_target, visibility_observation, attempt->match.T_target_source);
    }
    attempt->production_kept_initial_pose =
        attempt->initial_visibility.valid &&
        (!attempt->refined_visibility.valid ||
         attempt->initial_visibility.consistency_ratio >
             attempt->refined_visibility.consistency_ratio + 1e-9);
    attempt->production_pose = attempt->production_kept_initial_pose
                                   ? attempt->initial_pose
                                   : attempt->match.T_target_source;
  };

  result.oracle_nearest_keyframe_id = findNearestKeyframe(oracle_pose);
  RegistrationSeedProbeAttempt oracle_attempt;
  oracle_attempt.seed_kind = "oracle_gt";
  oracle_attempt.seed_keyframe_id = result.oracle_nearest_keyframe_id;
  oracle_attempt.initial_pose = oracle_pose;
  oracle_attempt.match =
      matcher_.alignPrepared(target, prepared_query, oracle_pose);
  finish_attempt(&oracle_attempt);
  result.attempts.push_back(std::move(oracle_attempt));

  const auto candidates = place_index_.search(query_cloud);
  if (candidates.empty()) {
    result.error = "no_descriptor_candidates";
    result.valid = true;
    return result;
  }
  const auto &candidate = candidates.front();
  result.descriptor_candidate_keyframe_id = candidate.match_id;
  const auto keyframe = keyframe_manager_.getKeyframe(candidate.match_id);
  if (!keyframe) {
    result.error = "descriptor_keyframe_missing";
    result.valid = true;
    return result;
  }

  const auto yaw_hypotheses = buildRelocalizationYawHypotheses(candidate, config_);
  for (const double yaw : yaw_hypotheses) {
    RegistrationSeedProbeAttempt attempt;
    attempt.seed_kind = "descriptor";
    attempt.seed_keyframe_id = candidate.match_id;
    attempt.yaw_offset_rad = yaw;
    attempt.initial_pose = keyframe->pose_optimized;
    attempt.initial_pose.linear() =
        keyframe->pose_optimized.linear() *
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    attempt.match =
        matcher_.alignPrepared(target, prepared_query, attempt.initial_pose);
    finish_attempt(&attempt);
    result.attempts.push_back(std::move(attempt));
  }
  result.valid = true;
  return result;
}

} // namespace n3mapping
