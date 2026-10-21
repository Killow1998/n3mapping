#include "n3mapping/world_localizing.h"
#include "relocalization_observation.h"
#include "relocalization_seed_plan.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <Eigen/Eigenvalues>
#include <glog/logging.h>

namespace n3mapping {
std::vector<const WorldLocalizing::RelocHypothesis*>
WorldLocalizing::confirmRelocalization(
    const PointCloudT::Ptr& query_cloud,
    const std::vector<LoopCandidate>& current_candidates,
    const Eigen::Isometry3d& odom_pose, RelocalizationObservation& observation,
    RelocalizationPerformance& cost, bool reloc_debug_enabled) {
  std::vector<const RelocHypothesis*> currently_registered;
  const auto& prepared_query = observation.registration;
  const auto& visibility_observation = observation.visibility;
  const auto hypothesis_visibility = [&](const PreparedRelocTarget &target,
                                         const Eigen::Isometry3d &pose) {
    const auto started = std::chrono::steady_clock::now();
    auto visibility = evaluatePreparedVisibilityConsistency(
        *target.visibility_target, visibility_observation, pose, target.visibility_indices.get());
    cost.visibility_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    ++cost.visibility_evaluations;
    return visibility;
  };

  auto descriptor_supports_keyframe =
      [&](const std::vector<LoopCandidate> &candidates, int64_t keyframe_id) {
        auto target = keyframe_manager_.getKeyframe(keyframe_id);
        if (!target) {
          return false;
        }
        const Eigen::Vector2d target_xy =
            target->pose_optimized.translation().head<2>();
        for (const auto &candidate : candidates) {
          auto candidate_kf = keyframe_manager_.getKeyframe(candidate.match_id);
          if (!candidate_kf) {
            continue;
          }
          const double d_xy =
              (candidate_kf->pose_optimized.translation().head<2>() - target_xy)
                  .norm();
          if (d_xy <= kRelocBasinAssignRadiusXY) {
            return true;
          }
        }
        return false;
      };

  hypothesis_manager_.advanceWindow();
  // Hypotheses are independent until ranking. Evaluate those using the same
  // immutable map target together, without reordering their stored identities
  // or retaining an unbounded set of prepared targets across observations.
  std::map<int64_t, std::vector<RelocHypothesis*>> target_groups;
  for (auto &hyp : hypothesis_manager_.hypotheses()) {
    if (!hyp.alive)
      continue;
    // This region was already evaluated against this observation at admission.
    // Counting it again would manufacture a second confirmation from one scan.
    if (hyp.first_observation == hypothesis_manager_.windowCount())
      continue;
    if (reloc_debug_enabled) {
      // This field describes a registration produced by this query only.
      // A miss must be null instead of silently reusing the prior query's H.
      hyp.last_registration_observability = {};
    }

    Eigen::Isometry3d predicted_pose = hyp.T_map_odom * odom_pose;
    int64_t nearest_kf_id = findNearestKeyframe(predicted_pose);
    if (nearest_kf_id < 0) {
      hyp.cumulative_log_likelihood -= config_.reloc_hypothesis_miss_penalty;
      continue;
    }

    if (!hyp.tracking_proposal &&
        !descriptor_supports_keyframe(current_candidates, nearest_kf_id)) {
      hyp.cumulative_log_likelihood -= config_.reloc_hypothesis_miss_penalty;
      continue;
    }
    target_groups[nearest_kf_id].push_back(&hyp);
  }

  for (const auto &[nearest_kf_id, hypotheses] : target_groups) {
    RelocTargetRequest target_request = makeRelocTargetRequest(nearest_kf_id);
    const PreparedRelocTarget prepared_target =
        target_provider_->getTarget(target_request);
    cost.target_build_ms += prepared_target.metrics.target_build_ms;
    cost.target_prepare_ms += prepared_target.metrics.target_prepare_ms;
    cost.target_builds += prepared_target.metrics.target_builds;
    cost.target_preparations += prepared_target.metrics.target_preparations;
    cost.cache_hits += prepared_target.metrics.cache_hit;
    cost.cache_misses += prepared_target.metrics.cache_miss;
    cost.effective_cache_max_bytes = prepared_target.metrics.effective_max_bytes;
    cost.effective_cache_max_entries = prepared_target.metrics.effective_max_entries;
    for (auto *hypothesis : hypotheses) {
      auto &hyp = *hypothesis;
      const Eigen::Isometry3d predicted_pose = hyp.T_map_odom * odom_pose;
      ++cost.hypotheses;
      if (!prepared_target.hasVisibility()) {
        hyp.cumulative_log_likelihood -= config_.reloc_hypothesis_miss_penalty;
        continue;
      }

      // Registration proposes a local correction; the LiDAR observation model
      // decides whether that correction actually explains the measured nearest
      // surface on each ray. This prevents one-way ICP from dragging a valid
      // pose onto a remote dense surface.
      observation.prepare(query_cloud, matcher_, config_, cost);
      const auto predicted_visibility =
          hypothesis_visibility(prepared_target, predicted_pose);
      MatchResult mr;
      MatchResult checked_prediction;
      bool optimized = false;
      if (prepared_target.valid()) {
        // Intermediate observations confirm the odometry-propagated pose
        // with fresh residuals. The final observation always refines it;
        // failed geometric confirmation falls back immediately as well.
        const int observations = hypothesis_manager_.windowCount() - hyp.first_observation + 1;
        if (observations < config_.reloc_temporal_window_size) {
          const auto started = std::chrono::steady_clock::now();
          mr = matcher_.evaluatePreparedPose(*prepared_target.registration_target,
                                             prepared_query, predicted_pose, hyp.metric);
          cost.fixed_pose_ms += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started).count();
          ++cost.fixed_pose_evaluations;
          ++cost.confirmation_evaluations;
          checked_prediction = mr;
        }
        if (!evaluateRelocMatchQuality(mr, config_).accepted ||
            (hyp.tracking_proposal && mr.inlier_ratio < config_.loop_min_inlier_ratio)) {
          const auto align_started = std::chrono::steady_clock::now();
          auto setting = matcher_.getSettings();
          if (hyp.tracking_proposal) {
            setting.type = hyp.metric.type;
            setting.downsampling_resolution = hyp.metric.resolution;
            setting.max_correspondence_distance = hyp.metric.max_correspondence_distance;
            setting.max_iterations = config_.gicp_max_iterations + config_.icp_refine_max_iterations;
          }
          // An accepted hypothesis already names its capture scale. Re-solve
          // the current observation there; a failed or large correction must
          // still receive the original broad search below.
          mr = hyp.tracking_proposal
              ? matcher_.alignPrepared(*prepared_target.registration_target,
                                       prepared_query, predicted_pose, setting)
              : matcher_.resumePrepared(*prepared_target.registration_target,
                                        prepared_query, predicted_pose, hyp.metric);
          cost.align_ms += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - align_started).count();
          ++cost.alignments;
          ++cost.confirmation_registrations;
          optimized = true;
          if (!hasValidRegistrationResult(mr)) ++cost.rejected_alignments;
          if (!hyp.tracking_proposal &&
              (!evaluateRelocMatchQuality(mr, config_).accepted ||
               (predicted_pose.inverse() * mr.T_target_source).translation().norm() >
                   config_.icp_refine_max_correspondence_distance ||
               Eigen::AngleAxisd(predicted_pose.rotation().transpose() *
                                mr.T_target_source.rotation()).angle() >
                   config_.reloc_track_max_rotation)) {
            const auto fallback_started = std::chrono::steady_clock::now();
            mr = matcher_.alignPrepared(*prepared_target.registration_target,
                                        prepared_query, predicted_pose);
            cost.align_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - fallback_started).count();
            ++cost.alignments;
            ++cost.confirmation_registrations;
            if (!hasValidRegistrationResult(mr)) ++cost.rejected_alignments;
          }
        }
      }
      VisibilityConsistencyResult selected_visibility = predicted_visibility;
      const bool current_registration_converged = optimized && mr.converged;
      Eigen::Isometry3d selected_pose = predicted_pose;
      bool selected_registration_endpoint = false;
      if (optimized && hasValidRegistrationResult(mr)) {
        const auto refined_visibility =
            hypothesis_visibility(prepared_target, mr.T_target_source);
        if (refined_visibility.valid &&
            (!selected_visibility.valid ||
             refined_visibility.consistency_ratio >
                 selected_visibility.consistency_ratio + 1e-9)) {
          selected_visibility = refined_visibility;
          selected_pose = mr.T_target_source;
          selected_registration_endpoint = true;
        }
        if (!selected_registration_endpoint) {
          // These are identical inputs within this observation, not evidence
          // from an older scan. A change of metric requires a fresh check.
          if (checked_prediction.termination == MatchTermination::FixedPoseEvaluation &&
              checked_prediction.metric.type == mr.metric.type &&
              checked_prediction.metric.resolution == mr.metric.resolution &&
              checked_prediction.metric.max_correspondence_distance ==
                  mr.metric.max_correspondence_distance) {
            mr = checked_prediction;
          } else {
            const auto fixed_started = std::chrono::steady_clock::now();
            mr = matcher_.evaluatePreparedPose(*prepared_target.registration_target,
                                               prepared_query, selected_pose,
                                               mr.metric);
            cost.fixed_pose_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - fixed_started).count();
            ++cost.fixed_pose_evaluations;
          }
        }
      }
      if (hyp.tracking_proposal &&
          (!evaluateRelocMatchQuality(mr, config_).accepted ||
           mr.inlier_ratio < config_.loop_min_inlier_ratio ||
           (predicted_pose.inverse() * selected_pose).translation().norm() >
               hyp.metric.max_correspondence_distance ||
           Eigen::AngleAxisd(predicted_pose.rotation().transpose() *
                            selected_pose.rotation()).angle() > config_.reloc_track_max_rotation ||
           !selected_visibility.valid ||
           selected_visibility.consistency_ratio < config_.loop_min_inlier_ratio)) {
        hyp.alive = false;
        continue;
      }
      hyp.cumulative_log_likelihood +=
          computeTrackLogLikelihood(mr, predicted_pose);
      hyp.num_updates += 1;
      if (hasValidRegistrationResult(mr)) {
        if (current_registration_converged && evaluateRelocMatchQuality(mr, config_).accepted)
          currently_registered.push_back(&hyp);
        hyp.T_map_odom = selected_pose * odom_pose.inverse();
        hyp.metric = mr.metric;
        hyp.last_match_id = nearest_kf_id;
        hyp.converged_updates += 1;

        // Recorded here, not at alignment, so the numbers describe the pose
        // this hypothesis actually carries.
        const bool information_at_selected_pose =
            selected_pose.isApprox(mr.T_target_source);
        const Eigen::Matrix3d trans_info = mr.information.block<3, 3>(0, 0);
        const Eigen::Matrix3d rot_info = mr.information.block<3, 3>(3, 3);
        const Eigen::Matrix3d cross_info = mr.information.block<3, 3>(3, 0);
        const Eigen::Matrix3d rot_marginal =
            rot_info -
            cross_info *
                trans_info.completeOrthogonalDecomposition().pseudoInverse() *
                cross_info.transpose();
        const Eigen::Vector3d re =
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(rot_info)
                .eigenvalues();
        const Eigen::Vector3d rm =
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(rot_marginal)
                .eigenvalues();
        const Eigen::Vector3d te =
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(trans_info)
                .eigenvalues();
        const bool production_quality =
            mr.success && mr.fitness_score <= config_.gicp_fitness_threshold &&
            mr.inlier_ratio >= config_.reloc_min_inlier_ratio;
        hyp.last_rot_info_min = re(0);
        hyp.last_rot_info_marginal_min = rm(0);
        hyp.last_trans_info_min = te(0);
        hyp.last_iterations = static_cast<int>(mr.iterations);
        hyp.last_termination = static_cast<int>(mr.termination);
        hyp.last_inlier_ratio = mr.inlier_ratio;
        hyp.last_fitness = mr.fitness_score;
        hyp.last_pose_is_refined = selected_registration_endpoint;
        hyp.last_production_quality = production_quality;
        if (reloc_debug_enabled) {
          hyp.last_registration_observability =
              analyzeRegistrationObservability(
                  mr,
                  selected_registration_endpoint ? "registration_endpoint"
                                                 : "motion_prediction",
                  information_at_selected_pose, production_quality);
        }
        VLOG(1) << "[Reloc/Info] seed=" << hyp.seed_match_id << " pose="
                << (selected_registration_endpoint ? "refined" : "predicted")
                << " prod_quality=" << production_quality
                << " rot_eig=" << re(0) << "," << re(1) << "," << re(2)
                << " rot_marg_eig=" << rm(0) << "," << rm(1) << "," << rm(2)
                << " trans_eig=" << te(0) << "," << te(1) << "," << te(2)
                << " inliers=" << mr.num_inliers
                << " fitness=" << mr.fitness_score
                << " inlier_ratio=" << mr.inlier_ratio;
      }
      if (selected_visibility.valid) {
        hyp.visibility_consistency_sum += selected_visibility.consistency_ratio;
        hyp.visibility_evidence_sum += selected_visibility.evidence_log_odds;
        hyp.visibility_updates += 1;
      }
    }
  }
  return currently_registered;
}
} // namespace n3mapping
