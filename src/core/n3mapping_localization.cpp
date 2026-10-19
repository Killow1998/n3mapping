// Localization input lifecycle, backend orchestration and offline probes.
#include "n3mapping/core/n3mapping_core.h"

namespace n3mapping {

core::BackendOutput
N3MappingCore::processLocalizationFrame(const core::LioFrame &frame) {
  auto &localizer = session_->worldLocalizing();
  if (!frame.pose_valid || !isFiniteRigidPose(frame.T_world_lidar)) {
    // A correction and its temporal hypotheses belong to one valid odometry
    // source. Invalid input cannot carry them across a frontend restart.
    localizer.resetLocalizationState();
    localization_source_frame_id_.clear();
    return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
  }
  if (localization_source_frame_id_ != frame.source_frame_id) {
    localizer.resetLocalizationState();
    localization_source_frame_id_ = frame.source_frame_id;
  }
  if (!frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
    return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  if (!map_loaded_) {
    return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  Eigen::Isometry3d pose_map = frame.T_world_lidar;
  bool success = false;
  bool relocalization_locked = false;
  RelocalizationState relocalization_state = RelocalizationState::SEARCHING;
  PoseSource pose_source = PoseSource::NONE;
  std::string relocalization_decision = "not_attempted";
  RelocalizationPerformance relocalization_performance;
  TrackingPerformance tracking_performance;
  int64_t seed_keyframe_id = -1;
  int64_t support_keyframe_id = -1;
  int64_t matched_keyframe_id = -1;
  bool tracking_attempted = false;

  if (localizer.isRelocalized()) {
    tracking_attempted = true;
    auto result = localizer.trackLocalization(frame.undistorted_cloud,
                                              frame.T_world_lidar);
    tracking_performance = result.tracking_performance;
    relocalization_state = result.state;
    pose_source = result.pose_source;
    relocalization_decision = result.decision;
    seed_keyframe_id = result.seed_keyframe_id;
    support_keyframe_id = result.support_keyframe_id;
    matched_keyframe_id = result.matched_keyframe_id;
    if (result.success) {
      pose_map = result.pose_in_map;
      success = true;
    }
  }

  if (!localizer.isRelocalized()) {
    const double query_timestamp =
        static_cast<double>(frame.stamp.nsec) * 1e-9;
    auto result = localizer.relocalize(
        frame.undistorted_cloud, frame.T_world_lidar, query_timestamp);
    relocalization_performance = result.performance;
    relocalization_decision = result.decision;
    relocalization_state = result.state;
    pose_source = result.pose_source;
    if (result.success) {
      pose_map = result.pose_in_map;
      success = true;
      relocalization_locked = true;
      seed_keyframe_id = result.seed_keyframe_id;
      support_keyframe_id = result.support_keyframe_id;
      matched_keyframe_id = result.matched_keyframe_id;
    }
  }

  if (!success) {
    pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
  }

  auto output = makeOutput(success, pose_map, frame.undistorted_cloud);
  output.relocalization_locked = relocalization_locked;
  output.relocalization_state = relocalization_state;
  output.pose_source = pose_source;
  output.relocalization_decision = relocalization_decision;
  output.relocalization_performance = relocalization_performance;
  output.tracking_performance = tracking_performance;
  if (relocalization_performance.available)
    output.performance.initial_relocalization_ms = relocalization_performance.total_ms;
  if (tracking_performance.available)
    output.performance.loaded_map_tracking_ms = tracking_performance.tracking_total_ms;
  output.relocalization_seed_keyframe_id = seed_keyframe_id;
  output.relocalization_support_keyframe_id = support_keyframe_id;
  output.matched_keyframe_id = matched_keyframe_id;
  output.performance.tracking_attempted = tracking_attempted;
  output.performance.enabled = config_.reloc_debug_enable;
  return output;
}

RegistrationSeedProbeResult N3MappingCore::probeLocalizationRegistration(
    const core::LioFrame::PointCloud::Ptr &cloud,
    const Eigen::Isometry3d &odom_pose, const Eigen::Isometry3d &oracle_pose) {
  if (!map_loaded_) {
    RegistrationSeedProbeResult result;
    result.error = "map_not_loaded";
    return result;
  }
  return session_->worldLocalizing().probeRegistrationSeeds(cloud, odom_pose,
                                                            oracle_pose);
}

} // namespace n3mapping
