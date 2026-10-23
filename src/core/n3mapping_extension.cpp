#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/n3mapping_session.h"
#include <chrono>

namespace n3mapping {

core::BackendOutput
N3MappingCore::processMapExtensionFrame(const core::LioFrame &frame) {
  TrackingPerformance tracking_performance;
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
    output.tracking_performance = tracking_performance;
    if (performance.enabled) {
      output.performance = performance;
    }
    return output;
  };

  if (!frame.pose_valid || !frame.undistorted_cloud ||
      frame.undistorted_cloud->empty() || !map_loaded_) {
    return make_output(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  auto &resuming = session_->mappingResuming();
  auto &localizer = session_->worldLocalizing();
  const auto state = resuming.getState();

  if (state == MappingResumingState::MAP_LOADED) {
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

  // While old-map geometry is visible, extension must track that immutable
  // geometry continuously. A constant relocalization transform only carries
  // the session LIO's time-varying drift into the resumed map. This strict
  // path never falls back to a new global search or to self-created frames.
  // Loss suppresses output and graph writes, but does not disable subsequent
  // loaded-map registration when usable geometry returns.
  performance.tracking_attempted = true;
  const auto tracking_started = timingStarted();
  const RelocResult tracking = localizer.trackLoadedMap(
      frame.undistorted_cloud, frame.T_world_lidar);
  tracking_performance = tracking.tracking_performance;
  if (performance.enabled) {
    performance.loaded_map_tracking_ms =
        elapsedMilliseconds(tracking_started);
  }
  if (!tracking.success ||
      tracking.state != RelocalizationState::FULL_6DOF_LOCKED ||
      tracking.pose_source != PoseSource::GEOMETRICALLY_CORRECTED ||
      tracking.matched_keyframe_id < 0) {
    auto output = make_output(
        false, tracking.pose_in_map, frame.undistorted_cloud);
    output.relocalization_state = tracking.state;
    output.pose_source = tracking.pose_source;
    output.relocalization_decision = tracking.decision;
    output.relocalization_seed_keyframe_id = tracking.seed_keyframe_id;
    output.relocalization_support_keyframe_id =
        tracking.support_keyframe_id;
    output.matched_keyframe_id = tracking.matched_keyframe_id;
    return output;
  }

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
      auto matched_kf = session_->keyframeManager().getKeyframe(
          tracking.matched_keyframe_id);
      if (matched_kf && matched_kf->is_from_loaded_map) {
        appendDenseTrajectorySample(timestamp, pose_map, matched_kf->id,
                                    matched_kf->pose_optimized, false);
      }
    }
    auto output = make_output(true, pose_map, frame.undistorted_cloud);
    output.relocalization_state = tracking.state;
    output.pose_source = tracking.pose_source;
    output.relocalization_decision = tracking.decision;
    output.relocalization_seed_keyframe_id = tracking.seed_keyframe_id;
    output.relocalization_support_keyframe_id =
        tracking.support_keyframe_id;
    output.matched_keyframe_id = tracking.matched_keyframe_id;
    return output;
  }

  const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
  MappingResumingTiming resuming_timing;
  int64_t reused_acquisition_id = -1;
  const int64_t keyframe_id = resuming.processNewKeyframe(
      timestamp, frame.T_world_lidar, frame.undistorted_cloud,
      tracking.matched_keyframe_id, tracking.pose_in_map,
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
    // The strict tracker already supplied a loaded-map constraint for this
    // keyframe (the first one is represented by SESSION_ANCHOR). Running the
    // global descriptor pipeline again would be redundant and can only add a
    // less local alias.
    const auto refresh_started = timingStarted();
    refreshOptimizedPoses();
    if (performance.enabled) {
      performance.post_commit_refresh_ms =
          elapsedMilliseconds(refresh_started);
    }
    // Publish the same Atlas-verified pose as on non-keyframes. The new graph
    // node remains an estimate for joint optimization, not tracking authority.
  }

  auto output = make_output(keyframe_id >= 0 || reused_acquisition_id >= 0, pose_map,
                            frame.undistorted_cloud);
  output.accepted_keyframe = keyframe_id >= 0;
  output.keyframe_id = keyframe_id >= 0 ? keyframe_id : reused_acquisition_id;
  output.relocalization_state = tracking.state;
  output.pose_source = tracking.pose_source;
  output.relocalization_decision = tracking.decision;
  output.relocalization_seed_keyframe_id = tracking.seed_keyframe_id;
  output.relocalization_support_keyframe_id = tracking.support_keyframe_id;
  output.matched_keyframe_id = tracking.matched_keyframe_id;
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
