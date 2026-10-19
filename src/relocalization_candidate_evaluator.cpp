#include "n3mapping/relocalization_candidate_evaluator.h"
#include "n3mapping/relocalization_order.h"
#include "n3mapping/relocalization_quality.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <glog/logging.h>

#include <pcl/common/transforms.h>

#include "n3mapping/pcl_compat.h"
#include "n3mapping/relocalization_state.h"

namespace n3mapping {
namespace {
bool sameSeed(const std::pair<int64_t, double>& lhs,
              const std::pair<int64_t, double>& rhs) {
  return lhs.first == rhs.first &&
         std::abs(std::remainder(lhs.second - rhs.second, 2.0 * M_PI)) <
             kDescriptorYawMergeTolerance;
}
} // namespace

RelocalizationCandidateEvaluator::RelocalizationCandidateEvaluator(
    const Config &config, KeyframeManager &keyframe_manager,
    PointCloudMatcher &matcher, RelocTargetProvider &target_provider)
    : config_(config), keyframe_manager_(keyframe_manager), matcher_(matcher),
      target_provider_(target_provider) {}

void RelocalizationCandidateEvaluator::clear() {
  std::lock_guard<std::mutex> lock(seed_mutex_);
  seed_poses_.clear();
  previous_seeds_.clear();
  seed_revision_ = {};
}

void RelocalizationCandidateEvaluator::beginSeedPlan(
    const std::vector<std::pair<int64_t, double>>& seeds,
    const KeyframeMapRevision& revision) {
  std::lock_guard<std::mutex> lock(seed_mutex_);
  if (!(seed_revision_ == revision)) {
    seed_poses_.clear();
    previous_seeds_.clear();
  }
  seed_revision_ = revision;
  for (auto it = seed_poses_.begin(); it != seed_poses_.end();) {
    const auto matches = [&](const auto& seed) { return sameSeed(seed, it->first); };
    if (!std::any_of(seeds.begin(), seeds.end(), matches) &&
        !std::any_of(previous_seeds_.begin(), previous_seeds_.end(), matches))
      it = seed_poses_.erase(it);
    else
      ++it;
  }
  previous_seeds_ = seeds;
}

std::size_t RelocalizationCandidateEvaluator::discardSeedsNear(
    const Eigen::Vector3d& center, double radius_xy) {
  std::lock_guard<std::mutex> lock(seed_mutex_);
  const auto before = seed_poses_.size();
  for (auto it = seed_poses_.begin(); it != seed_poses_.end();) {
    const auto anchor = keyframe_manager_.getKeyframe(it->first.first);
    if (!anchor || (anchor->pose_optimized.translation().head<2>() -
                    center.head<2>()).norm() <= radius_xy)
      it = seed_poses_.erase(it);
    else
      ++it;
  }
  return before - seed_poses_.size();
}

std::vector<RelocalizationCandidateEvaluation>
RelocalizationCandidateEvaluator::evaluate(
    const PointCloudMatcher::PreparedSource &prepared_query,
    const LoopCandidate &candidate, RelocTargetRequest target_request,
    const PreparedVisibilityObservation &visibility_observation,
    const Eigen::Isometry3d &odom_pose,
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
    auto result = evaluatePreparedVisibilityConsistency(*target, visibility_observation, pose,
        prepared_target.visibility_indices.get());
    cost.visibility_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    ++cost.visibility_evaluations;
    return result;
  };
  for (const double yaw : yaw_hypotheses) {
    const auto seed = std::make_pair(candidate.match_id, yaw);
    auto stored_seed = seed;
    Eigen::Isometry3d initial_pose = match_keyframe->pose_optimized;
    initial_pose.linear() =
        match_keyframe->pose_optimized.linear() *
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Isometry3d prediction;
    MatchMetric previous_metric;
    bool continuing = false;
    bool needs_registration = false;
    {
      std::lock_guard<std::mutex> lock(seed_mutex_);
      // Generator deduplication already treats these tiny angular differences
      // as the same seed. Exact floating keys otherwise miss neighboring SC
      // sectors calculated through different float/double expressions.
      const auto previous = std::find_if(seed_poses_.begin(), seed_poses_.end(),
          [&](const auto& entry) { return sameSeed(seed, entry.first); });
      if (previous != seed_poses_.end()) {
        stored_seed = previous->first;
        prediction = previous->second.map_from_odom * odom_pose;
        previous_metric = previous->second.metric;
        needs_registration = previous->second.needs_registration;
        continuing = true;
      }
    }
    MatchResult match;
    if (continuing && config_.reloc_temporal_window_size > 1) {
      ++cost.seed_continuations;
      const auto started = std::chrono::steady_clock::now();
      if (needs_registration) {
        ++cost.seed_resumptions;
        match = matcher_.alignPrepared(
            *prepared_target.registration_target, prepared_query, prediction);
        cost.align_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        ++cost.alignments;
      } else {
        match = matcher_.evaluatePreparedPose(
            *prepared_target.registration_target, prepared_query, prediction, previous_metric);
        cost.fixed_pose_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        ++cost.fixed_pose_evaluations;
      }
    }
    // Failure of an old mode must not suppress a new mode with the same seed.
    if (!evaluateRelocMatchQuality(match, config_).accepted) {
      ++cost.seed_initializations;
      if (continuing) ++cost.seed_retries;
      continuing = false;
      const auto align_started = std::chrono::steady_clock::now();
      match = matcher_.alignPrepared(
          *prepared_target.registration_target, prepared_query, initial_pose);
      cost.align_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - align_started).count();
      ++cost.alignments;
    }
    const bool production_quality =
        isFiniteRigidPose(match.T_target_source) &&
        (match.converged || (continuing &&
         match.termination == MatchTermination::FixedPoseEvaluation)) &&
        std::isfinite(match.fitness_score) &&
        std::isfinite(match.inlier_ratio) &&
        match.fitness_score < config_.gicp_fitness_threshold &&
        match.inlier_ratio >= config_.reloc_min_inlier_ratio;
    if (!production_quality) {
      ++cost.rejected_alignments;
      VLOG(1) << "[RelocSeedReject] anchor=" << candidate.match_id << " yaw=" << yaw
              << " termination=" << matchTerminationName(match.termination)
              << " iterations=" << match.iterations << " fitness=" << match.fitness_score
              << " inlier_ratio=" << match.inlier_ratio;
      std::lock_guard<std::mutex> lock(seed_mutex_);
      const auto quality = evaluateRelocMatchQuality(match, config_);
      // An unfinished solve may carry a useful proposal, but no evidence.
      // Its next use must converge on the current observation before admission.
      if (config_.reloc_temporal_window_size > 1 &&
          match.termination == MatchTermination::MaxIterations &&
          isFiniteRigidPose(match.T_target_source) && quality.fitness_pass &&
          quality.inlier_pass && quality.confidence_pass)
        seed_poses_[stored_seed] = {match.T_target_source * odom_pose.inverse(), match.metric, true};
      else
        seed_poses_.erase(stored_seed);
      continue;
    }

    const auto initial_visibility = visibility(initial_pose);
    auto selected_visibility =
        visibility(match.T_target_source);
    if (initial_visibility.valid &&
        (!selected_visibility.valid ||
         initial_visibility.consistency_ratio >
             selected_visibility.consistency_ratio + 1e-9)) {
      // ICP is only a local pose proposal. Keep the descriptor pose when it
      // explains the measured first-return geometry better than the optimizer's
      // endpoint.
      const auto fixed_started = std::chrono::steady_clock::now();
      match = matcher_.evaluatePreparedPose(
          *prepared_target.registration_target, prepared_query, initial_pose,
          match.metric);
      cost.fixed_pose_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - fixed_started).count();
      ++cost.fixed_pose_evaluations;
      selected_visibility = initial_visibility;
    }

    {
      std::lock_guard<std::mutex> lock(seed_mutex_);
      if (config_.reloc_temporal_window_size > 1 &&
          evaluateRelocMatchQuality(match, config_).accepted)
        seed_poses_[stored_seed] = {match.T_target_source * odom_pose.inverse(), match.metric};
      else
        seed_poses_.erase(stored_seed);
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
