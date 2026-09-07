#include "n3mapping/relocalization_candidate_evaluator.h"
#include "n3mapping/relocalization_order.h"

#include <algorithm>
#include <cmath>

#include <pcl/common/transforms.h>

#include "n3mapping/pcl_compat.h"
#include "n3mapping/relocalization_state.h"

namespace n3mapping {

RelocalizationCandidateEvaluator::RelocalizationCandidateEvaluator(
    const Config &config, KeyframeManager &keyframe_manager,
    PointCloudMatcher &matcher, RelocTargetProvider &target_provider)
    : config_(config), keyframe_manager_(keyframe_manager), matcher_(matcher),
      target_provider_(target_provider) {}

std::vector<RelocalizationCandidateEvaluation>
RelocalizationCandidateEvaluator::evaluate(
    const PointCloudT::Ptr &query_cloud,
    const PointCloudMatcher::PreparedSource &prepared_query,
    const LoopCandidate &candidate, RelocTargetRequest target_request) {
  std::vector<RelocalizationCandidateEvaluation> evaluations;
  auto match_keyframe = keyframe_manager_.getKeyframe(candidate.match_id);
  if (!match_keyframe) {
    return evaluations;
  }

  PointCloudT::Ptr target = target_request.local_target;
  if (!target || target->empty()) {
    if (!match_keyframe->cloud || match_keyframe->cloud->empty()) {
      return evaluations;
    }
    target = pcl::make_shared<PointCloudT>();
    const Eigen::Matrix4f transform =
        match_keyframe->pose_optimized.matrix().cast<float>();
    pcl::transformPointCloud(*match_keyframe->cloud, *target, transform);
  }

  target_request.local_target = target;
  const PreparedRelocTarget prepared_target =
      target_provider_.getTarget(target_request);
  if (!prepared_target.valid()) {
    return evaluations;
  }
  target = prepared_target.visibility_target;

  const auto yaw_hypotheses = buildDescriptorYawHypotheses(candidate, config_);
  for (const double yaw : yaw_hypotheses) {
    Eigen::Isometry3d initial_pose = match_keyframe->pose_optimized;
    initial_pose.linear() =
        match_keyframe->pose_optimized.linear() *
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const auto initial_visibility =
        evaluateVisibility(target, query_cloud, initial_pose);

    MatchResult match = matcher_.alignPrepared(
        *prepared_target.registration_target, prepared_query, initial_pose);
    const bool production_quality =
        isFiniteRigidPose(match.T_target_source) && match.converged &&
        std::isfinite(match.fitness_score) &&
        std::isfinite(match.inlier_ratio) &&
        match.fitness_score < config_.gicp_fitness_threshold &&
        match.inlier_ratio >= config_.reloc_min_inlier_ratio;
    if (!production_quality) {
      continue;
    }

    auto selected_visibility =
        evaluateVisibility(target, query_cloud, match.T_target_source);
    if (initial_visibility.valid &&
        (!selected_visibility.valid ||
         initial_visibility.consistency_ratio >
             selected_visibility.consistency_ratio + 1e-9)) {
      // ICP is only a local pose proposal. Keep the descriptor pose when it
      // explains the measured first-return geometry better than the optimizer's
      // endpoint.
      match = matcher_.evaluatePreparedPose(
          *prepared_target.registration_target, prepared_query, initial_pose,
          match.metric);
      selected_visibility = initial_visibility;
    }

    RelocalizationCandidateEvaluation evaluation;
    evaluation.match = std::move(match);
    evaluation.matched_keyframe_id = candidate.match_id;
    evaluation.visibility = selected_visibility;
    evaluations.push_back(std::move(evaluation));
  }

  std::sort(evaluations.begin(), evaluations.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.visibility.valid != rhs.visibility.valid) {
                return lhs.visibility.valid;
              }
              if (lhs.visibility.valid) {
                const int order = compareRelocalizationScore(
                    lhs.visibility.consistency_ratio, rhs.visibility.consistency_ratio);
                if (order != 0) return order < 0;
              }
              const int quality_order = compareRelocalizationScore(
                  -lhs.match.fitness_score, -rhs.match.fitness_score);
              if (quality_order != 0) return quality_order < 0;
              if (lhs.matched_keyframe_id != rhs.matched_keyframe_id)
                return lhs.matched_keyframe_id < rhs.matched_keyframe_id;
              return relocalizationPoseLess(lhs.match.T_target_source,
                                            rhs.match.T_target_source);
            });

  // Different yaw seeds often converge onto the same physical solution. Keep
  // one representative per mode, but retain genuinely different orientations
  // so temporal ambiguity logic can reject unobservable symmetries.
  std::vector<RelocalizationCandidateEvaluation> distinct;
  distinct.reserve(evaluations.size());
  for (const auto &evaluation : evaluations) {
    const bool duplicate =
        std::any_of(distinct.begin(), distinct.end(), [&](const auto &kept) {
          const Eigen::Isometry3d delta =
              kept.match.T_target_source.inverse() * evaluation.match.T_target_source;
          return delta.translation().norm() <
                     config_.reloc_ambiguity_min_basin_separation &&
                 Eigen::AngleAxisd(delta.rotation()).angle() <
                     config_.reloc_track_max_rotation;
        });
    if (!duplicate) {
      distinct.push_back(evaluation);
    }
  }
  return distinct;
}

VisibilityConsistencyResult
RelocalizationCandidateEvaluator::evaluateVisibility(
    const PointCloudT::Ptr &target_cloud, const PointCloudT::Ptr &query_cloud,
    const Eigen::Isometry3d &T_map_lidar) const {
  if (!target_cloud || !query_cloud) {
    return {};
  }
  return evaluateVisibilityConsistency(*target_cloud, *query_cloud, T_map_lidar,
                                       visibilityOptionsFromConfig(config_));
}

} // namespace n3mapping
