#include "n3mapping/relocalization_debug_logger.h"

#include "n3mapping/world_localizing.h"
#include "n3mapping/tracking_target_cache.h"
#include "n3mapping/visibility_consistency.h"
#include "registration_planar_geometry.h"
#include "tracking_pose_correction.h"
#include "tracking_geometry.h"
#include "point_cloud_target_view.h"
#include "registration_search.h"

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
    RelocTrackingDebugEvent& debug_event, WeakTranslationAxes& weak_axes,
    const std::optional<Eigen::Vector3d>& previous_proposal, Eigen::Vector3d& confirmed_shift) {
  using Clock = std::chrono::steady_clock;
  const auto timing_started = [] { return Clock::now(); };
  const auto elapsed_ms = [](Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  };
  const auto registration_started = timing_started();
  MatchResult match_result;
  if (prepared_target) {
    if (config_.icp_refine_use_gicp && consecutive_track_failures_ == 0) {
      const auto coverage = matcher_.evaluatePreparedPose(
          *prepared_target, prepared_source, predicted_pose,
          {small_gicp::RegistrationSetting::PLANE_ICP,
           config_.gicp_downsampling_resolution,
           config_.gicp_max_correspondence_distance});
      if (evaluateRelocMatchQuality(coverage, config_).accepted) {
        auto local = matcher_.getSettings();
        local.type = small_gicp::RegistrationSetting::GICP;
        local.downsampling_resolution = config_.icp_refine_downsampling_resolution;
        local.max_correspondence_distance = config_.icp_refine_max_correspondence_distance;
        local.max_iterations = config_.icp_refine_max_iterations;
        auto proposal = matcher_.alignPrepared(
            *prepared_target, prepared_source, predicted_pose, local);
        const auto delta = predicted_pose.inverse() * proposal.T_target_source;
        if (evaluateRelocMatchQuality(proposal, config_).accepted &&
            delta.translation().norm() <= config_.icp_refine_max_correspondence_distance &&
            Eigen::AngleAxisd(delta.rotation()).angle() <= config_.reloc_track_max_rotation)
          match_result = std::move(proposal);
      }
    }
    if (!hasValidRegistrationResult(match_result))
      match_result = matcher_.alignPrepared(
          *prepared_target, prepared_source, predicted_pose);
  }
  debug_event.registration_ms = elapsed_ms(registration_started);
  bool retry_used = false;
  bool bounded_recovery_accepted = false;
  std::shared_ptr<const PointCloudMatcher::PreparedTarget> recovery_pair_target;
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
         std::min(config_.icp_refine_max_correspondence_distance,
                  candidate.metric.max_correspondence_distance)});
    debug_event.retry_registration_ms += elapsed_ms(started);
    const bool supported = hasValidRegistrationResult(coverage) &&
        coverage.fitness_score < config_.gicp_fitness_threshold &&
        coverage.inlier_ratio >= config_.loop_min_inlier_ratio;
    VLOG(1) << "Tracking recovery map coverage=" << coverage.inlier_ratio
            << " fitness=" << coverage.fitness_score
            << " accepted=" << supported;
    if (!supported) return false;

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
    bool visible = visibility.valid && visibility.consistency_ratio >=
                                           config_.loop_min_inlier_ratio;
    if (!visible && visibility.valid && visibility.evidence_log_odds > 0.0) {
      const auto geometry_started = timing_started();
      const auto geometry = matcher_.evaluatePreparedPlaneGeometry(
          localization_atlas_->preparedTarget(), prepared_source,
          candidate.T_target_source,
          {small_gicp::RegistrationSetting::PLANE_ICP,
           config_.gicp_downsampling_resolution,
           std::min(config_.gicp_downsampling_resolution,
                    candidate.metric.max_correspondence_distance)});
      const double planar_information = minimumPlanarRegistrationInformation(
          geometry.information,
          candidate.T_target_source.rotation().transpose() *
              Eigen::Vector3d::UnitZ());
      visible = hasValidRegistrationResult(geometry) &&
                planar_information >= config_.gicp_num_neighbors;
      debug_event.retry_registration_ms += elapsed_ms(geometry_started);
      VLOG(1) << "Tracking recovery stable-geometry information="
              << planar_information << " accepted=" << visible;
    }
    VLOG(1) << "Tracking recovery first-return consistency="
            << visibility.consistency_ratio
            << " known_fraction=" << visibility.known_fraction
            << " consistent_given_known=" << visibility.consistent_given_known
            << " conflict_given_known="
            << visibility.foreground_conflict_given_known
            << " accepted=" << visible;
    return visible;
  };

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
      prepared_target = &recovery_target;
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
      if (use_atlas && config_.gicp_downsampling_resolution <
                           local.max_correspondence_distance &&
          (!hasValidRegistrationResult(recovery) ||
          recovery.fitness_score >= config_.gicp_fitness_threshold ||
          recovery.inlier_ratio < config_.reloc_min_inlier_ratio)) {

        auto narrow = local;
        narrow.max_correspondence_distance = config_.gicp_downsampling_resolution;
        const auto started = timing_started();
        auto checked = matcher_.alignPrepared(recovery_target, prepared_source,
                                               predicted_pose, narrow);
        debug_event.retry_registration_ms += elapsed_ms(started);
        if (!hasValidRegistrationResult(checked) ||
            checked.fitness_score >= config_.gicp_fitness_threshold ||
            checked.inlier_ratio < config_.loop_min_inlier_ratio) {
          const auto verification_started = timing_started();
          checked = matcher_.evaluatePreparedPose(
              recovery_target, prepared_source, predicted_pose,
              {small_gicp::RegistrationSetting::GICP,
               config_.icp_refine_downsampling_resolution,
               narrow.max_correspondence_distance});
          debug_event.retry_registration_ms += elapsed_ms(verification_started);
        }
        const auto change = predicted_pose.inverse() * checked.T_target_source;
        if (change.translation().norm() <= narrow.max_correspondence_distance &&
            admit_recovery(checked)) recovery = std::move(checked);
      }
      if ((!hasValidRegistrationResult(recovery) ||
           recovery.fitness_score >= config_.gicp_fitness_threshold ||
           recovery.inlier_ratio < config_.reloc_min_inlier_ratio) &&
          debug_event.nearest_kf_id >= 0 && localization_atlas_ &&
          localization_atlas_->loaded()) {

        const auto pair_started = timing_started();
        const auto pair_target = tracking_targets_->acquireLocal(debug_event.nearest_kf_id, 0);
        const auto pair = pair_target.target
            ? matcher_.alignPrepared(*pair_target.target, prepared_source, predicted_pose, local)
            : MatchResult{};
        debug_event.retry_registration_ms += elapsed_ms(pair_started);
        if (admit_recovery(pair)) {
          recovery = pair;
          recovery_pair_target = pair_target.target;
          prepared_target = recovery_pair_target.get();
        }
      }

      match_result = std::move(recovery);
    }
  }
  if (prepared_target && prepared_source.has_refine_cloud &&
      hasValidRegistrationResult(match_result)) {
    const auto projection_started = timing_started();
    const auto supported_pose = supportedTrackingPose(
        *prepared_source.refine_cloud, config_.gicp_num_neighbors,
        predicted_pose, match_result, weak_axes);
    if (!supported_pose) {
      match_result = MatchResult{};
    } else if ((supported_pose->matrix() - match_result.T_target_source.matrix()).norm() > 1e-9) {
      debug_event.registration_translation_projected = true;

      auto checked = matcher_.evaluatePreparedPose(
          *prepared_target, prepared_source, *supported_pose, match_result.metric);
      checked.stages = std::move(match_result.stages);
      match_result = std::move(checked);
    }
    debug_event.registration_ms += elapsed_ms(projection_started);
  }
  if (!strict_loaded_map && weak_axes && weak_axes->size() == 1 &&
      localization_atlas_ && localization_atlas_->loaded() &&
      localization_atlas_->preparedTarget().has_refine_level &&
      hasValidRegistrationResult(match_result)) {
    const auto started = timing_started();
    const auto &level = localization_atlas_->preparedTarget().refine_level;
    const auto &source = *prepared_source.refine_cloud;
    const Eigen::Vector3d axis = weak_axes->front();
    const auto &endpoint = match_result.T_target_source;
    const Eigen::Vector3d local_axis = endpoint.rotation().transpose() * axis;
    std::vector<Eigen::Vector4d> points;
    for (std::size_t i = 0; i < source.size(); ++i) {
      if (local_axis.dot(source.cov(i).topLeftCorner<3, 3>() * local_axis) < 0.5)
        points.push_back(endpoint * source.point(i));
    }
    const double radius = config_.icp_refine_max_correspondence_distance;
    const double spacing = config_.gicp_downsampling_resolution;
    const int steps = static_cast<int>(config_.reloc_track_max_translation / spacing);
    double center_support = 0;
    const auto propose = [&](const auto &tree, const auto &surface) -> double {
      const auto score = [&](int step) {
        const Eigen::Vector3d shift = axis * (step * spacing);
        double total = 0.0;
        for (const auto &point : points) {
          auto query = point;
          query.head<3>() += shift;
          std::size_t i;
          double distance_sq;
          if (tree.nearest(query, &i, &distance_sq) &&
              std::abs(axis.dot(surface.normal(i).template head<3>())) > std::sqrt(0.5))
            total += 1.0 - distance_sq / (radius * radius);
        }
        return total;
      };
      const double center = score(0);
      center_support = center;
      std::vector<double> scores(2 * steps + 1);
      for (int step = -steps; step <= steps; ++step)
        scores[step + steps] = step ? score(step) : center;
      const auto peak = std::max_element(scores.begin(), scores.end());
      const int best_step = static_cast<int>(peak - scores.begin()) - steps;
      const int same_peak_steps = static_cast<int>(std::lround(2 * radius / spacing));
      if (std::abs(best_step) <= same_peak_steps) return 0.0;
      double alternative = center;
      for (int step = -steps; step <= steps; ++step)
        if (std::abs(step - best_step) > same_peak_steps)
          alternative = std::max(alternative, scores[step + steps]);
      return *peak - alternative >= config_.gicp_num_neighbors ? best_step * spacing : 0.0;
    };
    if (points.size() >= static_cast<std::size_t>(config_.gicp_num_neighbors)) {
      using namespace registration_detail;
      const double shift = level.mapped
          ? propose(MappedCorrespondenceSearch{*level.mapped, radius * radius, level.use_occupied_bounds},
                    MappedTargetView{*level.mapped})
          : propose(CorrespondenceSearch{*level.kdtree, radius * radius, level.search_bounds.get()},
                    TargetView{*level.cloud, level.lazy_normals.get()});
      if (shift != 0.0) {
        auto seed = endpoint;
        seed.translation() += axis * shift;
        auto local = matcher_.getSettings();
        local.type = small_gicp::RegistrationSetting::GICP;
        local.downsampling_resolution = config_.icp_refine_downsampling_resolution;
        local.max_correspondence_distance = radius;
        local.max_iterations = config_.icp_refine_max_iterations;
        auto recovered = matcher_.alignPrepared(*prepared_target, prepared_source, seed, local);
        const auto delta = seed.inverse() * recovered.T_target_source;
        if (evaluateRelocMatchQuality(recovered, config_).accepted &&
            delta.translation().norm() <= radius &&
            (predicted_pose.inverse() * recovered.T_target_source).translation().norm() <= config_.reloc_track_max_translation &&
            Eigen::AngleAxisd(delta.rotation()).angle() <= config_.reloc_track_max_rotation) {
          WeakTranslationAxes recovered_axes;
          auto pose = supportedTrackingPose(source, config_.gicp_num_neighbors, seed, recovered, recovered_axes);
          if (pose) {
            auto checked = matcher_.evaluatePreparedPose(*prepared_target, prepared_source, *pose, recovered.metric);
            if (evaluateRelocMatchQuality(checked, config_).accepted) {
              const Eigen::Vector3d proposal = axis * axis.dot(
                  checked.T_target_source.translation() - endpoint.translation());
              if (previous_proposal && (proposal - *previous_proposal).norm() <= radius) {
                const auto support_gain = [&](const auto &tree) {
                  double gain = 0.0;
                  const double limit = radius * radius;
                  for (std::size_t i = 0; i < source.size(); ++i) {
                    const auto &point = source.point(i);
                    std::size_t nearest;
                    double here, there;
                    const bool hit_here = tree.nearest(endpoint * point, &nearest, &here);
                    const bool hit_there = tree.nearest(*pose * point, &nearest, &there);
                    gain += (hit_here ? here : limit) - (hit_there ? there : limit);
                  }
                  return gain;
                };
                const double gain = level.mapped
                    ? support_gain(MappedCorrespondenceSearch{*level.mapped, radius * radius, level.use_occupied_bounds})
                    : support_gain(CorrespondenceSearch{*level.kdtree, radius * radius, level.search_bounds.get()});
                if (gain > 0.0) {
                  confirmed_shift = proposal;
                  match_result = std::move(checked);
                  weak_axes = std::move(recovered_axes);
                  VLOG(1) << "Directional recovery shift=" << shift << " center_support=" << center_support;
                }
                VLOG(1) << "Whole scan support gain=" << gain << " proposal=" << shift;
              } else {
                directional_proposal_ = proposal;
              }
            }
          }
        }
      }
    }
    const double search_ms = elapsed_ms(started);
    debug_event.registration_ms += search_ms;
    VLOG(1) << "Directional search ms=" << search_ms << " points=" << points.size() << " center=" << center_support;
  }
  debug_event.icp_converged = match_result.converged;
  debug_event.fitness_score = match_result.fitness_score;
  debug_event.inlier_ratio = match_result.inlier_ratio;
  debug_event.retry_used = retry_used;
  return match_result;
}
}
