// Tracking registration proposals and bounded recovery. No localization
// state or map-to-odometry correction is committed in this translation unit.
#include "n3mapping/world_localizing.h"

#include <chrono>
#include <glog/logging.h>

namespace n3mapping {
MatchResult WorldLocalizing::registerTracking(
    const PointCloudMatcher::PreparedTarget* prepared_target,
    const PointCloudMatcher::PreparedSource& prepared_source,
    const Eigen::Isometry3d& predicted_pose, bool strict_loaded_map,
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

  // If ICP failed with standard params and we have recent failures, retry with
  // wider search
  const bool icp_failed =
      !hasValidRegistrationResult(match_result) ||
      match_result.fitness_score >= config_.gicp_fitness_threshold ||
      match_result.inlier_ratio < config_.reloc_min_inlier_ratio;
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
    debug_event.retry_registration_ms = elapsed_ms(retry_started);
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
      proposal_has_geometry) {
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
