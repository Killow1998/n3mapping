#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/n3mapping_session.h"
#include "n3mapping/async_log.h"
#include <chrono>

namespace n3mapping {

bool N3MappingCore::startMapExtension(std::string* error) {
  const auto reject = [&](const char* reason) {
    if (error) *error = reason;
    logEvent("map_extension_switch_rejected", reason);
    return false;
  };
  if (!map_loaded_) return reject("map_not_loaded");
  auto& resuming = session_->mappingResuming();
  const auto state = resuming.getState();
  if (state == MappingResumingState::RELOCALIZED ||
      state == MappingResumingState::EXTENDING) return true;
  if (!extension_anchor_frame_) return reject("localization_not_authoritative");
  // Seed the existing odometry check with the last verified observation, so
  // the first extension frame is checked against the same continuous source.
  if (mappingInputBlockReason(*extension_anchor_frame_) != core::MappingBlockReason::None)
    return reject("invalid_extension_anchor_input");
  if (!resuming.adoptCurrentLocalization(extension_anchor_frame_->source_frame_id))
    return reject("extension_anchor_unavailable");
  extension_anchor_frame_.reset();
  logEvent("map_extension_started", "reused_localization=true");
  return true;
}

core::BackendOutput
N3MappingCore::processMapExtensionFrame(const core::LioFrame &frame) {
  extension_anchor_frame_.reset();
  RelocResult tracking;
  using PerformanceClock = std::chrono::steady_clock;
  core::BackendPerformanceTiming performance;
  performance.enabled = config_.reloc_debug_enable;
  const auto timingStarted = [&]() {
    return performance.enabled ? PerformanceClock::now()
                               : PerformanceClock::time_point{};
  };
  const auto elapsedMilliseconds = [&](PerformanceClock::time_point started) {
    return std::chrono::duration<double, std::milli>(PerformanceClock::now() -
                                                     started)
        .count();
  };
  const auto make_output = [&](bool success, const Eigen::Isometry3d &pose,
                               const PointCloud::Ptr &cloud) {
    auto output = makeOutput(success, pose, cloud);
    if (performance.tracking_attempted) {
      output.tracking_performance = tracking.tracking_performance;
      output.relocalization_state = tracking.state;
      output.pose_source = tracking.pose_source;
      output.relocalization_decision = tracking.decision;
      output.relocalization_seed_keyframe_id = tracking.seed_keyframe_id;
      output.relocalization_support_keyframe_id = tracking.support_keyframe_id;
      output.matched_keyframe_id = tracking.matched_keyframe_id;
    }
    if (performance.enabled) {
      output.performance = performance;
    }
    return output;
  };

  auto &resuming = session_->mappingResuming();
  auto &localizer = session_->worldLocalizing();
  const auto state = resuming.getState();
  // An established anchor belongs to one LIO coordinate frame. Never connect
  // a changed source frame to that session's odometry chain.
  const bool source_changed = state != MappingResumingState::MAP_LOADED &&
      localization_source_frame_id_ != frame.source_frame_id;
  const auto blocked = !map_loaded_ || source_changed
      ? core::MappingBlockReason::InvalidInput : mappingInputBlockReason(frame);
  if (blocked != core::MappingBlockReason::None) {
    auto output = make_output(false, frame.T_world_lidar, frame.undistorted_cloud);
    output.mapping_block_reason = blocked;
    if (source_changed) output.relocalization_decision = "map_extension_source_frame_changed";
    return output;
  }

  if (state == MappingResumingState::MAP_LOADED) {
    if (localization_source_frame_id_ != frame.source_frame_id) {
      localizer.reset();
      localization_source_frame_id_ = frame.source_frame_id;
    }
    const auto relocalization_started = timingStarted();
    RelocalizationPerformance relocalization_performance;
    const bool locked = resuming.performInitialRelocalization(
        frame.undistorted_cloud, frame.T_world_lidar,
        frame.source_frame_id, &relocalization_performance);
    if (performance.enabled) {
      performance.initial_relocalization_ms =
          elapsedMilliseconds(relocalization_started);
    }
    auto output = make_output(
        locked, localizer.getMapToOdomTransform() * frame.T_world_lidar,
        frame.undistorted_cloud);
    output.relocalization_locked = locked;
    output.relocalization_performance = relocalization_performance;
    output.relocalization_state =
        locked ? RelocalizationState::FULL_6DOF_LOCKED
               : RelocalizationState::SEARCHING;
    output.pose_source =
        locked ? PoseSource::GEOMETRICALLY_CORRECTED : PoseSource::NONE;
    output.relocalization_decision =
        locked ? "accepted" : "initial_relocalization_rejected";
    if (locked) {
      const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
      const int64_t matched_id = localizer.getLastMatchedKeyframeId();
      output.relocalization_seed_keyframe_id = matched_id;
      output.relocalization_support_keyframe_id = matched_id;
      output.matched_keyframe_id = matched_id;
      auto matched_kf = session_->keyframeManager().getKeyframe(matched_id);
      if (!external_dense_trajectory_recording_enabled_ && matched_kf) {
        appendDenseTrajectorySample(timestamp, output.T_world_lidar, matched_id,
                                    matched_kf->pose_optimized, false);
      }
    }
    return output;
  }

  if (state != MappingResumingState::RELOCALIZED &&
      state != MappingResumingState::EXTENDING) {
    return make_output(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  // Loaded geometry supplies absolute corrections. When it disappears, the
  // anchored session can still extend with LIO odometry, without inventing a
  // loaded-map constraint or granting global localization authority.
  performance.tracking_attempted = true;
  const auto tracking_started = timingStarted();
  tracking = localizer.trackLoadedMap(frame.undistorted_cloud, frame.T_world_lidar);
  if (performance.enabled) {
    performance.loaded_map_tracking_ms = elapsedMilliseconds(tracking_started);
  }
  const bool tracked = tracking.success &&
      tracking.state == RelocalizationState::FULL_6DOF_LOCKED &&
      tracking.pose_source == PoseSource::GEOMETRICALLY_CORRECTED &&
      tracking.matched_keyframe_id >= 0;
  if (!tracked) tracking.pose_source = PoseSource::ODOM_PREDICTED;

  Eigen::Isometry3d pose_map = tracking.pose_in_map;
  const auto keyframe_gate_started = timingStarted();
  const bool should_add_keyframe =
      resuming.shouldAddKeyframe(frame.T_world_lidar);
  if (performance.enabled) {
    performance.keyframe_gate_ms =
        elapsedMilliseconds(keyframe_gate_started);
  }
  if (!should_add_keyframe) {
    if (!external_dense_trajectory_recording_enabled_) {
      const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
      auto anchor = tracked
          ? session_->keyframeManager().getKeyframe(tracking.matched_keyframe_id)
          : session_->keyframeManager().getLatestKeyframe();
      if (anchor) {
        const bool loaded = anchor->is_from_loaded_map;
        appendDenseTrajectorySample(timestamp, loaded ? pose_map : frame.T_world_lidar,
                                    anchor->id, loaded ? anchor->pose_optimized : anchor->pose_odom,
                                    false);
      }
    }
    return make_output(true, pose_map, frame.undistorted_cloud);
  }

  const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
  MappingResumingTiming resuming_timing;
  int64_t reused_acquisition_id = -1;
  const int64_t keyframe_id = resuming.processNewKeyframe(
      timestamp, frame.T_world_lidar, frame.undistorted_cloud,
      tracked ? tracking.matched_keyframe_id : -1, tracking.pose_in_map,
      performance.enabled ? &resuming_timing : nullptr, &reused_acquisition_id);
  if (performance.enabled) {
    performance.keyframe_commit_ms = resuming_timing.total_ms;
    performance.graph_update_ms = resuming_timing.graph_update_ms;
    performance.descriptor_update_ms = resuming_timing.descriptor_update_ms;
  }
  if (keyframe_id >= 0) {
    if (config_.loop_geometry_enable) {
      std::lock_guard<std::mutex> lock(loop_queue_mutex_);
      loop_detection_queue_.push_back(keyframe_id);
    }
    // Return to old geometry is handled by the same strict local tracker.
    // Joint geometry remains asynchronous; no second global search is needed.
    const auto refresh_started = timingStarted();
    refreshOptimizedPoses();
    if (performance.enabled) {
      performance.post_commit_refresh_ms =
          elapsedMilliseconds(refresh_started);
    }
    // Keep the observation pose and its quality together. A graph solve
    // does not turn an odometry prediction into geometric tracking authority.
  }

  auto output = make_output(keyframe_id >= 0 || reused_acquisition_id >= 0, pose_map,
                            frame.undistorted_cloud);
  output.accepted_keyframe = keyframe_id >= 0;
  output.keyframe_id = keyframe_id >= 0 ? keyframe_id : reused_acquisition_id;
  if (!external_dense_trajectory_recording_enabled_ && keyframe_id >= 0) {
    if (auto kf = session_->keyframeManager().getKeyframe(keyframe_id)) {
      appendDenseTrajectorySample(timestamp, kf->pose_odom, keyframe_id,
                                  kf->pose_odom, false);
    }
  } else if (!external_dense_trajectory_recording_enabled_ && reused_acquisition_id >= 0) {
    if (auto anchor = session_->keyframeManager().getKeyframe(tracking.matched_keyframe_id)) {
      appendDenseTrajectorySample(timestamp, pose_map, anchor->id,
                                  anchor->pose_optimized, false);
    }
  }
  return output;
}

} // namespace n3mapping
