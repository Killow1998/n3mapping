#include "n3mapping/world_localizing.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <glog/logging.h>
#include <iterator>
#include <set>
#include "n3mapping/tracking_target_cache.h"
#include "tracking_pose_correction.h"

namespace n3mapping {
RelocResult
WorldLocalizing::trackLocalization(const PointCloudT::Ptr &cloud,
                                   const Eigen::Isometry3d &odom_pose) {
  return trackLocalizationImpl(cloud, odom_pose, false);
}

RelocResult
WorldLocalizing::trackLoadedMap(const PointCloudT::Ptr &cloud,
                                const Eigen::Isometry3d &odom_pose) {
  return trackLocalizationImpl(cloud, odom_pose, true);
}

RelocResult
WorldLocalizing::trackLocalizationImpl(const PointCloudT::Ptr &cloud,
                                       const Eigen::Isometry3d &odom_pose,
                                       bool strict_loaded_map) {
  using TrackingClock = std::chrono::steady_clock;
  RelocResult result;
  result.success = false;
  const bool reloc_debug_enabled = debug_emitter_.enabled();
  const auto tracking_started = TrackingClock::now();
  RelocTrackingDebugEvent debug_event = debug_emitter_.beginTracking();
  debug_event.available = true;
  debug_event.strict_loaded_map = strict_loaded_map;
  const auto timing_started = [&]() {
    return TrackingClock::now();
  };
  const auto elapsed_ms = [&](const TrackingClock::time_point &started) {
    return std::chrono::duration<double, std::milli>(TrackingClock::now() -
                                                     started)
        .count();
  };
  auto finish_tracking_debug = [&](const std::string &reject_reason) {
    result.decision = reject_reason.empty()
                          ? (strict_loaded_map ? "loaded_map_tracking_geometric"
                                               : "tracking_geometric")
                          : reject_reason;
    debug_event.tracking_total_ms = elapsed_ms(tracking_started);
    result.tracking_performance = debug_event;
    if (!reloc_debug_enabled) {
      return;
    }
    debug_emitter_.finishTracking(debug_event, result.success,
                                  consecutive_track_failures_, reject_reason);
  };

  if (!cloud || cloud->empty()) {
    std::lock_guard<std::mutex> lock(mutex_);
    result.state = is_relocalized_ ? RelocalizationState::RECENTLY_LOST
                                   : (has_ever_relocalized_
                                          ? RelocalizationState::LOST
                                          : RelocalizationState::SEARCHING);
    result.pose_source =
        is_relocalized_ ? PoseSource::ODOM_PREDICTED : PoseSource::NONE;
    result.seed_keyframe_id = relocalization_seed_id_;
    result.support_keyframe_id = last_matched_id_;
    result.matched_keyframe_id = last_matched_id_;
    result.pose_in_map = T_map_odom_ * odom_pose;
    debug_event.predicted_pose = result.pose_in_map;
    LOG(WARNING) << "Empty point cloud for tracking.";
    finish_tracking_debug("empty_cloud");
    return result;
  }

  const auto lock_started = timing_started();
  std::lock_guard<std::mutex> lock(mutex_);
  debug_event.lock_wait_ms = elapsed_ms(lock_started);
  result.seed_keyframe_id = relocalization_seed_id_;

  Eigen::Isometry3d predicted_pose = T_map_odom_ * odom_pose;
  result.pose_in_map = predicted_pose;
  debug_event.predicted_pose = predicted_pose;

  if (strict_loaded_map && !has_ever_relocalized_) {
    result.state = RelocalizationState::SEARCHING;
    result.pose_source = PoseSource::NONE;
    finish_tracking_debug("loaded_map_prior_missing");
    return result;
  }

  // Ordinary localization hands LOST back to its global search owner.
  // Extension has no such handoff: keep testing the loaded reference locally,
  // without publishing prediction-only poses or resetting the loss counter.
  if (!strict_loaded_map &&
      consecutive_track_failures_ > config_.reloc_max_track_failures) {
    is_relocalized_ = false;
    result.state = RelocalizationState::LOST;
    result.pose_source = PoseSource::NONE;
    finish_tracking_debug("max_track_failures");
    return result;
  }

  const auto nearest_keyframe_started = timing_started();
  int64_t nearest_kf_id = strict_loaded_map
                              ? findNearestLoadedKeyframe(predicted_pose)
                              : findNearestKeyframe(predicted_pose);
  debug_event.nearest_keyframe_ms = elapsed_ms(nearest_keyframe_started);
  debug_event.nearest_kf_id = nearest_kf_id;
  if (nearest_kf_id < 0) {
    consecutive_track_failures_++;
    if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
      is_relocalized_ = false;
      result.success = false;
      result.state = RelocalizationState::LOST;
      result.pose_source = PoseSource::NONE;
    } else {
      result.success = !strict_loaded_map;
      result.state = RelocalizationState::RECENTLY_LOST;
      result.pose_source = PoseSource::ODOM_PREDICTED;
    }
    result.matched_keyframe_id = last_matched_id_;
    result.support_keyframe_id = last_matched_id_;
    result.pose_in_map = predicted_pose;
    result.confidence = 0.5;
    last_odom_pose_ = odom_pose;
    finish_tracking_debug("nearest_keyframe_missing");
    return result;
  }

  // Build submap — use larger range when tracking is unstable. Extension
  // tracking must never use newly appended keyframes as its own reference.
  int submap_range = config_.gicp_submap_size;
  if (consecutive_track_failures_ > 0)
    submap_range =
        std::max(submap_range, config_.reloc_track_unstable_submap_size);

  PointCloudT::Ptr submap;
  TrackingTargetCache::TargetResult tracking_target;
  if (strict_loaded_map) {
    tracking_target = tracking_targets_->acquireLoaded(nearest_kf_id);
    submap = tracking_target.visibility_cloud;
    debug_event.loaded_map_cache_ms = tracking_target.loaded_map_cache_ms;
    debug_event.submap_build_ms = tracking_target.submap_build_ms;
  } else {
    // Keep the established ordinary-localization target unchanged. The
    // loaded-map-only target is specific to map-extension safety.
    tracking_target = tracking_targets_->acquireLocal(nearest_kf_id, submap_range);
    debug_event.submap_build_ms = tracking_target.submap_build_ms;
    if (VLOG_IS_ON(1)) {
      LocalMapSelectionRequest shadow_request;
      shadow_request.predicted_pose = predicted_pose;
      shadow_request.anchor_id = nearest_kf_id;
      shadow_request.spatial_radius =
          std::max(config_.keyframe_distance_threshold,
                   static_cast<double>(submap_range) *
                       config_.keyframe_distance_threshold);
      shadow_request.max_spatial_keyframes = 2 * submap_range + 1;
      // A frozen localization map has no active-session tail. Adding the
      // highest map ids would mix an unrelated map endpoint into the target;
      // session-aware tails belong after WP-10 introduces session identity.
      shadow_request.recent_tail_count = 0;
      shadow_request.map_revision = keyframe_manager_.revision();
      const LocalMapSelection shadow_selection =
          local_map_selector_.select(shadow_request);

      std::set<int64_t> legacy_ids;
      for (int64_t id = nearest_kf_id - submap_range;
           id <= nearest_kf_id + submap_range; ++id) {
        if (keyframe_manager_.getKeyframe(id)) {
          legacy_ids.insert(id);
        }
      }
      const std::set<int64_t> spatial_ids(shadow_selection.keyframe_ids.begin(),
                                          shadow_selection.keyframe_ids.end());
      std::size_t intersection_size = 0;
      for (const int64_t id : legacy_ids) {
        intersection_size += spatial_ids.count(id);
      }
      const std::size_t union_size =
          legacy_ids.size() + spatial_ids.size() - intersection_size;
      const double jaccard = union_size > 0
                                 ? static_cast<double>(intersection_size) /
                                       static_cast<double>(union_size)
                                 : 1.0;
      VLOG(1) << "[LocalMapShadow] path=tracking anchor_id=" << nearest_kf_id
              << " radius=" << shadow_request.spatial_radius
              << " max_spatial=" << shadow_request.max_spatial_keyframes
              << " recent_tail=0 legacy_size=" << legacy_ids.size()
              << " spatial_size=" << spatial_ids.size()
              << " jaccard=" << jaccard
              << " legacy_points=" << tracking_target.submap_points
              << " spatial_points=" << shadow_selection.selected_points
              << " index_rebuilt=" << shadow_selection.index_rebuilt
              << " index_build_ms=" << shadow_selection.index_build_ms
              << " query_ms=" << shadow_selection.query_ms << " revision_match="
              << shadow_selection.requested_revision_matched
              << " loaded=" << shadow_selection.loaded_keyframes
              << " current=" << shadow_selection.current_keyframes
              << " mixed_loaded_state="
              << (shadow_selection.loaded_keyframes > 0 &&
                  shadow_selection.current_keyframes > 0)
              << " z_span=" << (shadow_selection.z_max - shadow_selection.z_min)
              << " floor_labels=unavailable";
    }
  }
  debug_event.submap_size = tracking_target.submap_points;
  if (debug_event.submap_size == 0) {
    consecutive_track_failures_++;
    if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
      is_relocalized_ = false;
      result.success = false;
      result.state = RelocalizationState::LOST;
      result.pose_source = PoseSource::NONE;
    } else {
      result.success = !strict_loaded_map;
      result.state = RelocalizationState::RECENTLY_LOST;
      result.pose_source = PoseSource::ODOM_PREDICTED;
    }
    result.matched_keyframe_id = last_matched_id_;
    result.support_keyframe_id = last_matched_id_;
    result.pose_in_map = predicted_pose;
    result.confidence = 0.5;
    last_odom_pose_ = odom_pose;
    finish_tracking_debug("empty_submap");
    return result;
  }

  const auto *prepared_target = tracking_target.target.get();
  if (strict_loaded_map) {
    debug_event.loaded_map_target_cache_hit = tracking_target.hit;
    debug_event.loaded_map_target_cache_miss = tracking_target.miss;
  } else {
    debug_event.localization_target_cache_enabled = tracking_target.enabled;
    debug_event.localization_target_cache_hit = tracking_target.hit;
    debug_event.localization_target_cache_miss = tracking_target.miss;
    debug_event.localization_target_cache_entry_bytes = tracking_target.entry_bytes;
    debug_event.localization_target_cache_total_bytes = tracking_target.total_bytes;
    debug_event.localization_target_cache_entries = tracking_target.entries;
  }
  debug_event.target_prepare_ms = tracking_target.target_prepare_ms;
  const auto source_prepare_started = timing_started();
  const auto prepared_source = matcher_.prepareSourceCloud(cloud);
  debug_event.source_prepare_ms = elapsed_ms(source_prepare_started);
  auto match_result = registerTracking(prepared_target, prepared_source, *cloud,
                                       predicted_pose, odom_pose, strict_loaded_map, debug_event);
  const bool retry_used = debug_event.retry_used;

  const bool icp_ok =
      hasValidRegistrationResult(match_result) &&
      match_result.fitness_score < config_.gicp_fitness_threshold &&
      match_result.inlier_ratio >= config_.reloc_min_inlier_ratio;

  double delta_translation = std::numeric_limits<double>::infinity();
  double delta_rotation = std::numeric_limits<double>::infinity();
  double registration_delta_translation =
      std::numeric_limits<double>::infinity();
  double registration_delta_rotation = std::numeric_limits<double>::infinity();
  VisibilityConsistencyResult visibility;
  VisibilityConsistencyResult registration_visibility;
  if (hasValidRegistrationResult(match_result)) {
    const Eigen::Isometry3d registration_delta =
        predicted_pose.inverse() * match_result.T_target_source;
    registration_delta_translation = registration_delta.translation().norm();
    registration_delta_rotation =
        Eigen::AngleAxisd(registration_delta.rotation()).angle();
    debug_event.visibility_registration_delta_translation_m =
        registration_delta_translation;
    debug_event.visibility_registration_delta_rotation_rad =
        registration_delta_rotation;
    if (strict_loaded_map) {
      const auto visibility_started = timing_started();
      const auto visibility_observation = prepareVisibilityObservation(
          *cloud, visibilityOptionsFromConfig(config_));
      const bool endpoint_fast_enabled =
          config_.loaded_map_visibility_endpoint_fast_enable;
      debug_event.visibility_endpoint_fast_enabled = endpoint_fast_enabled;
      VisibilityConsistencyResult predicted_visibility;
      bool prediction_visibility_evaluated = false;
      const auto evaluate_prediction_visibility = [&]() {
        const auto prediction_visibility_started = timing_started();
        predicted_visibility = evaluatePreparedVisibilityConsistency(
            *submap, visibility_observation, predicted_pose);
        prediction_visibility_evaluated = true;
        if (reloc_debug_enabled) {
          debug_event.visibility_prediction_ms =
              elapsed_ms(prediction_visibility_started);
          debug_event.visibility_prediction_valid = predicted_visibility.valid;
          debug_event.visibility_prediction_consistency_ratio =
              predicted_visibility.consistency_ratio;
          debug_event.visibility_prediction_evidence_log_odds =
              predicted_visibility.evidence_log_odds;
        }
      };

      // Preserve the exact legacy evaluation order while the experiment is
      // disabled. Enabled runs evaluate the endpoint first so a fully accepted
      // endpoint can terminate the cascade.
      if (!endpoint_fast_enabled) {
        evaluate_prediction_visibility();
      }

      const auto registration_visibility_started = timing_started();
      registration_visibility = evaluatePreparedVisibilityConsistency(
          *submap, visibility_observation, match_result.T_target_source);
      if (reloc_debug_enabled) {
        debug_event.visibility_registration_ms =
            elapsed_ms(registration_visibility_started);
        debug_event.visibility_registration_valid =
            registration_visibility.valid;
        debug_event.visibility_registration_consistency_ratio =
            registration_visibility.consistency_ratio;
        debug_event.visibility_registration_evidence_log_odds =
            registration_visibility.evidence_log_odds;
      }

      const bool endpoint_fast_taken =
          endpoint_fast_enabled && icp_ok && !retry_used &&
          registration_delta_translation <=
              config_.reloc_track_max_translation &&
          registration_delta_rotation <= config_.reloc_track_max_rotation &&
          registration_visibility.valid &&
          registration_visibility.evidence_log_odds > 0.0;
      debug_event.visibility_endpoint_fast_taken = endpoint_fast_taken;
      if (endpoint_fast_enabled && !endpoint_fast_taken) {
        evaluate_prediction_visibility();
      }

      visibility = registration_visibility;
      debug_event.visibility_selected_pose_source = "registration_endpoint";
      if (prediction_visibility_evaluated && predicted_visibility.valid &&
          (!visibility.valid || predicted_visibility.consistency_ratio >
                                    visibility.consistency_ratio + 1e-9)) {
        // ICP is a local proposal, not authority to slide along a repeated
        // surface. Retain the prediction when the immutable loaded map
        // explains it better than the optimizer endpoint.
        auto predicted_match = matcher_.evaluatePreparedPose(
            *prepared_target, prepared_source, predicted_pose,
            match_result.metric);
        // Visibility ranks geometrically admissible poses. A prediction with
        // better ray agreement must not replace a valid endpoint if its own
        // residuals fail the very same quality gate.
        if (hasValidRegistrationResult(predicted_match) &&
            predicted_match.fitness_score < config_.gicp_fitness_threshold &&
            predicted_match.inlier_ratio >= config_.reloc_min_inlier_ratio) {
          match_result = std::move(predicted_match);
          visibility = predicted_visibility;
          debug_event.visibility_selected_pose_source = "motion_prediction";
        }
      }
      debug_event.visibility_ms = elapsed_ms(visibility_started);
    }
    const Eigen::Isometry3d delta =
        predicted_pose.inverse() * match_result.T_target_source;
    delta_translation = delta.translation().norm();
    delta_rotation = Eigen::AngleAxisd(delta.rotation()).angle();
  }

  debug_event.visibility_registration_would_accept =
      strict_loaded_map && icp_ok &&
      registration_delta_translation <= config_.reloc_track_max_translation &&
      registration_delta_rotation <= config_.reloc_track_max_rotation &&
      registration_visibility.valid &&
      registration_visibility.evidence_log_odds > 0.0;

  const bool strict_geometry_ok =
      delta_translation <= config_.reloc_track_max_translation &&
      delta_rotation <= config_.reloc_track_max_rotation &&
      (!strict_loaded_map ||
       (visibility.valid && visibility.evidence_log_odds > 0.0));

  // The proposal's convergence remains necessary, but its quality cannot
  // certify a different selected pose. Recompute confidence from that pose.
  const bool selected_quality_ok =
      hasValidRegistrationResult(match_result) &&
      match_result.fitness_score < config_.gicp_fitness_threshold &&
      match_result.inlier_ratio >= config_.reloc_min_inlier_ratio &&
      (strict_loaded_map || !config_.icp_refine_use_gicp ||
       match_result.metric.type == small_gicp::RegistrationSetting::GICP);
  const double scale = config_.gicp_fitness_threshold * 0.5;
  const double current_confidence =
      std::clamp(std::exp(-match_result.fitness_score / scale), 0.0, 1.0);
  debug_event.fitness_score = match_result.fitness_score;
  debug_event.inlier_ratio = match_result.inlier_ratio;
  const bool accepted_icp = icp_ok && selected_quality_ok && strict_geometry_ok;

  if (accepted_icp) {
    T_map_odom_ = trackingPoseCorrection(
        T_map_odom_, odom_pose, match_result.T_target_source,
        current_confidence, strict_loaded_map);

    result.success = true;
    result.state = RelocalizationState::FULL_6DOF_LOCKED;
    result.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
    result.matched_keyframe_id = nearest_kf_id;
    result.support_keyframe_id = nearest_kf_id;
    result.pose_in_map = T_map_odom_ * odom_pose;
    result.fitness_score = match_result.fitness_score;
    result.confidence = current_confidence;

    last_matched_id_ = nearest_kf_id;
    last_odom_pose_ = odom_pose;
    is_relocalized_ = true;
    has_ever_relocalized_ = true;
    consecutive_track_failures_ = 0;

    VLOG(1) << (strict_loaded_map ? "Loaded-map tracking OK: fitness="
                                  : "Tracking OK: fitness=")
            << match_result.fitness_score << " conf=" << current_confidence
            << " delta_t=" << delta_translation
            << " delta_r=" << delta_rotation
            << " visibility_logodds=" << visibility.evidence_log_odds;
    finish_tracking_debug("");
  } else {
    consecutive_track_failures_++;
    if (consecutive_track_failures_ <= 3 ||
        consecutive_track_failures_ % 5 == 0) {
      LOG(WARNING) << "Tracking failed (x" << consecutive_track_failures_
                   << "): optimizer_converged=" << debug_event.icp_converged
                   << " selected_pose="
                   << debug_event.visibility_selected_pose_source
                   << " evaluation="
                   << matchTerminationName(match_result.termination)
                   << " fitness=" << match_result.fitness_score
                   << " inlier=" << match_result.inlier_ratio
                   << " nearest_kf=" << nearest_kf_id
                   << " submap_pts=" << debug_event.submap_size
                   << " cloud_pts=" << cloud->size() << " pos=("
                   << predicted_pose.translation().x() << ","
                   << predicted_pose.translation().y() << ","
                   << predicted_pose.translation().z() << ")";
    }

    if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
      is_relocalized_ = false;
      result.success = false;
      result.state = RelocalizationState::LOST;
      result.pose_source = PoseSource::NONE;
    } else {
      // Keep odometry-based continuity for transient dropouts.
      result.success = !strict_loaded_map;
      result.state = RelocalizationState::RECENTLY_LOST;
      result.pose_source = PoseSource::ODOM_PREDICTED;
    }

    result.matched_keyframe_id = last_matched_id_;
    result.support_keyframe_id = last_matched_id_;
    result.pose_in_map = predicted_pose;
    result.confidence = 0.2;
    last_odom_pose_ = odom_pose;
    finish_tracking_debug(
        !icp_ok ? "icp_gate_failed"
                : (!selected_quality_ok ? "tracking_selected_pose_quality_gate_failed"
                                       : (strict_loaded_map
                                              ? "loaded_map_tracking_geometry_gate_failed"
                                              : "tracking_motion_gate_failed")));
  }

  return result;
}

} // namespace n3mapping
