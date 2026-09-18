#include "n3mapping/relocalization_candidate_evaluator.h"
#include "n3mapping/relocalization_order.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <glog/logging.h>

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
    const PointCloudMatcher::PreparedSource &prepared_query,
    const LoopCandidate &candidate, RelocTargetRequest target_request,
    const PreparedVisibilityObservation &visibility_observation,
    RelocalizationPerformance *performance) {
  RelocalizationPerformance unused_performance;
  auto &cost = performance ? *performance : unused_performance;
  cost.candidates = 1;
  std::vector<RelocalizationCandidateEvaluation> evaluations;
  auto match_keyframe = keyframe_manager_.getKeyframe(candidate.match_id);
  if (!match_keyframe) {
    return evaluations;
  }

  target_request.build_fallback_target = [match_keyframe]() {
    if (!match_keyframe->cloud || match_keyframe->cloud->empty()) {
      return PointCloudT::Ptr{};
    }
    auto target = pcl::make_shared<PointCloudT>();
    const Eigen::Matrix4f transform =
        match_keyframe->pose_optimized.matrix().cast<float>();
    pcl::transformPointCloud(*match_keyframe->cloud, *target, transform);
    return target;
  };
  const PreparedRelocTarget prepared_target =
      target_provider_.getTarget(target_request);
  cost.target_build_ms = prepared_target.metrics.target_build_ms;
  cost.target_prepare_ms = prepared_target.metrics.target_prepare_ms;
  cost.target_builds = prepared_target.metrics.target_builds;
  cost.target_preparations = prepared_target.metrics.target_preparations;
  cost.cache_hits = prepared_target.metrics.cache_hit;
  cost.cache_misses = prepared_target.metrics.cache_miss;
  cost.effective_cache_max_bytes = prepared_target.metrics.effective_max_bytes;
  cost.effective_cache_max_entries = prepared_target.metrics.effective_max_entries;
  if (!prepared_target.valid()) {
    return evaluations;
  }
  const auto &target = prepared_target.visibility_target;

  const auto yaw_hypotheses = buildDescriptorYawHypotheses(candidate, config_);
  const auto visibility = [&](const Eigen::Isometry3d &pose) {
    const auto start = std::chrono::steady_clock::now();
    auto result = evaluatePreparedVisibilityConsistency(*target, visibility_observation, pose);
    cost.visibility_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    ++cost.visibility_evaluations;
    return result;
  };
  for (const double yaw : yaw_hypotheses) {
    Eigen::Isometry3d initial_pose = match_keyframe->pose_optimized;
    initial_pose.linear() =
        match_keyframe->pose_optimized.linear() *
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const auto initial_visibility =
        visibility(initial_pose);

    const auto align_started = std::chrono::steady_clock::now();
    MatchResult match = matcher_.alignPrepared(
        *prepared_target.registration_target, prepared_query, initial_pose);
    cost.align_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - align_started).count();
    ++cost.alignments;
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
        visibility(match.T_target_source);
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
  VLOG(1) << "[RelocCandidateCost] anchor=" << candidate.match_id
          << " align_ms=" << cost.align_ms << " alignments=" << cost.alignments
          << " visibility_ms=" << cost.visibility_ms
          << " visibility_evaluations=" << cost.visibility_evaluations
          << " accepted_modes=" << distinct.size();
  return distinct;
}

} // namespace n3mapping
