// Tracking registration proposals and bounded recovery. No localization
// state or map-to-odometry correction is committed in this translation unit.
#include "n3mapping/world_localizing.h"
#include "n3mapping/tracking_target_cache.h"
#include "n3mapping/visibility_consistency.h"
#include "tracking_pose_correction.h"

#include <chrono>
#include <cmath>
#include <glog/logging.h>
#include <optional>

namespace n3mapping {
MatchResult WorldLocalizing::registerTracking(
    const PointCloudMatcher::PreparedTarget* prepared_target,
    const PointCloudMatcher::PreparedSource& prepared_source,
    const PointCloudT& query_cloud,
    const Eigen::Isometry3d& predicted_pose, const Eigen::Isometry3d& odom_pose,
    bool strict_loaded_map,
    RelocTrackingDebugEvent& debug_event) {
  using Clock = std::chrono::steady_clock;
  const auto timing_started = [] { return Clock::now(); };
  const auto elapsed_ms = [](Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  };
  const auto registration_started = timing_started();
  MatchResult match_result;
  if (prepared_target) {
    match_result =
        matcher_.alignPrepared(*prepared_target, prepared_source, predicted_pose);
  }
  debug_event.registration_ms = elapsed_ms(registration_started);
  bool retry_used = false;
  bool bounded_recovery_accepted = false;
  std::optional<PreparedVisibilityObservation> recovery_observation;
  const auto admit_recovery = [&](const MatchResult& candidate) {
    if (!hasValidRegistrationResult(candidate) ||
        candidate.metric.type != small_gicp::RegistrationSetting::GICP ||
        candidate.fitness_score >= config_.gicp_fitness_threshold ||
        candidate.inlier_ratio < config_.reloc_min_inlier_ratio ||
        !localization_atlas_ || !localization_atlas_->loaded())
      return false;
    const auto delta = predicted_pose.inverse() * candidate.T_target_source;
    if (delta.translation().norm() > config_.reloc_track_max_translation ||
        Eigen::AngleAxisd(delta.rotation()).angle() > config_.reloc_track_max_rotation)
      return false;
    const auto started = timing_started();
    const auto coverage = matcher_.evaluatePreparedPose(
        localization_atlas_->preparedTarget(), prepared_source, candidate.T_target_source,
        {small_gicp::RegistrationSetting::GICP,
         config_.icp_refine_downsampling_resolution,
         config_.icp_refine_max_correspondence_distance});
    debug_event.retry_registration_ms += elapsed_ms(started);
    const bool supported = hasValidRegistrationResult(coverage) &&
        coverage.fitness_score < config_.gicp_fitness_threshold &&
        coverage.inlier_ratio >= config_.loop_min_inlier_ratio;
    VLOG(1) << "Tracking recovery map coverage=" << coverage.inlier_ratio
            << " fitness=" << coverage.fitness_score
            << " accepted=" << supported;
    if (!supported) return false;
    // The optimizer endpoint already passed geometric quality and coverage.
    // Independently check the actual filtered body pose we would publish,
    // which can differ materially from the raw ICP endpoint.
    const double confidence = std::clamp(
        std::exp(-candidate.fitness_score / (config_.gicp_fitness_threshold * 0.5)),
        0.0, 1.0);
    const auto selected_pose = trackingPoseCorrection(
        T_map_odom_, odom_pose, candidate.T_target_source, confidence, false) * odom_pose;
    const auto visibility_started = timing_started();
    if (!recovery_observation) {
      recovery_observation = prepareVisibilityObservation(
          query_cloud, visibilityOptionsFromConfig(config_));
    }
    const auto indices = localization_atlas_->cropVisibilityIndices(
        selected_pose.translation());
    const auto visibility = evaluatePreparedVisibilityConsistency(
        *localization_atlas_->globalMap(), *recovery_observation,
        selected_pose, indices.get());
    debug_event.visibility_ms =
        (std::isfinite(debug_event.visibility_ms) ? debug_event.visibility_ms : 0.0) +
        elapsed_ms(visibility_started);
    const bool visible = visibility.valid &&
        visibility.consistency_ratio >= config_.loop_min_inlier_ratio;
    VLOG(1) << "Tracking recovery first-return consistency="
            << visibility.consistency_ratio << " accepted=" << visible;
    return visible;
  };

  // A failed coarse solve may have left the prediction's correspondence basin.
  // Only a fully verified fine result can replace the established wider retry.
  bool icp_failed =
      !hasValidRegistrationResult(match_result) ||
      match_result.fitness_score >= config_.gicp_fitness_threshold ||
      match_result.inlier_ratio < config_.reloc_min_inlier_ratio;
  if (icp_failed && !strict_loaded_map && prepared_target &&
      config_.icp_refine_use_gicp && localization_atlas_ &&
      localization_atlas_->loaded()) {
    auto bounded = matcher_.getSettings();
    bounded.max_correspondence_distance = config_.icp_refine_max_correspondence_distance;
    const auto started = timing_started();
    auto retry = matcher_.alignPrepared(*prepared_target, prepared_source, predicted_pose, bounded);
    debug_event.retry_registration_ms = elapsed_ms(started);
    retry_used = true;
    const auto delta = predicted_pose.inverse() * retry.T_target_source;
    if (delta.translation().norm() <= config_.icp_refine_max_correspondence_distance &&
        admit_recovery(retry)) {
      match_result = std::move(retry);
      bounded_recovery_accepted = true;
      icp_failed = false;
    }
  }
  if (icp_failed &&
      consecutive_track_failures_ < config_.reloc_track_retry_max_failures) {
    auto saved = matcher_.getSettings();
    auto wide = saved;
    wide.max_correspondence_distance = saved.max_correspondence_distance *
                                       config_.reloc_track_retry_corr_scale;
    wide.max_iterations = config_.reloc_track_retry_max_iterations;
    const auto retry_started = timing_started();
    auto retry = prepared_target
                     ? matcher_.alignPrepared(*prepared_target, prepared_source,
                                              predicted_pose, wide)
                     : MatchResult{};
    debug_event.retry_registration_ms =
        (retry_used ? debug_event.retry_registration_ms : 0.0) + elapsed_ms(retry_started);
    retry_used = true;
    if (hasValidRegistrationResult(retry) &&
        (!hasValidRegistrationResult(match_result) ||
         retry.fitness_score < match_result.fitness_score))
      match_result = retry;
  }
  if (icp_failed) {
    for (const auto& stage : match_result.stages) {
      VLOG(1) << "Tracking stage=" << stage.stage
              << " termination=" << matchTerminationName(stage.termination)
              << " iterations=" << stage.iterations
              << " fitness=" << stage.fitness_score
              << " inlier=" << stage.inlier_ratio;
    }
  }
  // A coarse solve can leave the prediction's correspondence basin and still
  // converge on a repeated structure. Recheck such proposals from the original
  // prediction. The loaded atlas includes geometry absent from a time-adjacent
  // keyframe submap, without building another target or using a wider radius.
  // Recovery refines an admissible proposal; it must not turn a failed broad
  // geometry check into new tracking authority in an already aliased region.
  const bool proposal_has_geometry =
      hasValidRegistrationResult(match_result) &&
      match_result.fitness_score < config_.gicp_fitness_threshold &&
      match_result.inlier_ratio >= config_.reloc_min_inlier_ratio;
  if (!strict_loaded_map && prepared_target && config_.icp_refine_use_gicp &&
      proposal_has_geometry && !bounded_recovery_accepted) {
    const auto proposal_delta = predicted_pose.inverse() * match_result.T_target_source;
    if (match_result.metric.type != small_gicp::RegistrationSetting::GICP ||
        proposal_delta.translation().norm() >
            config_.icp_refine_max_correspondence_distance ||
        Eigen::AngleAxisd(proposal_delta.rotation()).angle() >
            config_.reloc_track_max_rotation) {
      auto local = matcher_.getSettings();
      local.type = small_gicp::RegistrationSetting::GICP;
      local.downsampling_resolution = config_.icp_refine_downsampling_resolution;
      local.max_correspondence_distance = config_.icp_refine_max_correspondence_distance;
      local.max_iterations = config_.gicp_max_iterations + config_.icp_refine_max_iterations;
      const bool use_atlas = localization_atlas_ && localization_atlas_->loaded();
      const auto& recovery_target =
          use_atlas ? localization_atlas_->preparedTarget() : *prepared_target;
      const auto recovery_started = timing_started();
      auto recovery = matcher_.alignPrepared(
          recovery_target, prepared_source, predicted_pose, local);
      const double recovery_ms = elapsed_ms(recovery_started);
      debug_event.retry_registration_ms =
          (retry_used ? debug_event.retry_registration_ms : 0.0) + recovery_ms;
      retry_used = true;
      VLOG(1) << "Tracking local recovery: atlas=" << use_atlas
              << " rejected_delta_t="
              << proposal_delta.translation().norm() << " rejected_delta_r="
              << Eigen::AngleAxisd(proposal_delta.rotation()).angle()
              << " converged=" << recovery.converged
              << " fitness=" << recovery.fitness_score
              << " inlier=" << recovery.inlier_ratio
              << " iterations=" << recovery.iterations
              << " wall_ms=" << recovery_ms;
      if ((!hasValidRegistrationResult(recovery) ||
           recovery.fitness_score >= config_.gicp_fitness_threshold ||
           recovery.inlier_ratio < config_.reloc_min_inlier_ratio) &&
          debug_event.nearest_kf_id >= 0 && localization_atlas_ &&
          localization_atlas_->loaded()) {
        // An assembled map can blur surfaces through relative-pose uncertainty.
        // The original scan is a separate fine target, available only after the
        // broad submap has supplied geometric authority for this prediction.
        const auto pair_started = timing_started();
        const auto pair_target = tracking_targets_->acquireLocal(debug_event.nearest_kf_id, 0);
        const auto pair = pair_target.target
            ? matcher_.alignPrepared(*pair_target.target, prepared_source, predicted_pose, local)
            : MatchResult{};
        debug_event.retry_registration_ms += elapsed_ms(pair_started);
        if (admit_recovery(pair)) {
          recovery = pair;
        }
      }
      // An unsuccessful recheck cannot certify the distant coarse endpoint.
      // The caller applies the unchanged convergence, quality and motion gates.
      match_result = std::move(recovery);
    }
  }
  debug_event.icp_converged = match_result.converged;
  debug_event.fitness_score = match_result.fitness_score;
  debug_event.inlier_ratio = match_result.inlier_ratio;
  debug_event.retry_used = retry_used;
  return match_result;
}
} // namespace n3mapping
