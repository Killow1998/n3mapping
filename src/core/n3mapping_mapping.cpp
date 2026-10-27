#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/n3mapping_session.h"
#include "n3mapping/floor_attitude.h"
#include <glog/logging.h>

namespace n3mapping {

core::MappingBlockReason
N3MappingCore::mappingInputBlockReason(const core::LioFrame &frame) {
  if (!frame.pose_valid || !frame.undistorted_cloud ||
      frame.undistorted_cloud->empty() || !isFiniteRigidPose(frame.T_world_lidar)) {
    return core::MappingBlockReason::InvalidInput;
  }

  const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
  // Once the front end has run away nothing downstream can use what it sends,
  // and every frame that keeps being added replaces a usable partial map with
  // an unusable whole one. Stop taking them.
  if (config_.odom_sanity_enable) {
    if (!odometry_sanity_configured_) {
      OdometrySanityLimits limits;
      limits.max_speed_mps = config_.odom_sanity_max_speed_mps;
      limits.max_angular_rate_dps = config_.odom_sanity_max_angular_rate_dps;
      limits.max_consecutive_violations = config_.odom_sanity_max_consecutive;
      odometry_sanity_ = OdometrySanity(limits);
      odometry_sanity_configured_ = true;
    }
    if (odometry_sanity_.check(timestamp, frame.T_world_lidar).diverged) {
      return core::MappingBlockReason::OdometryDiverged;
    }
    if (odometry_sanity_.verdict().consecutive > 0) {
      return core::MappingBlockReason::InvalidInput;
    }
  }
  return core::MappingBlockReason::None;
}

core::BackendOutput
N3MappingCore::processMappingFrame(const core::LioFrame &frame) {
  const auto blocked = mappingInputBlockReason(frame);
  if (blocked != core::MappingBlockReason::None) {
    auto output = makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    output.mapping_block_reason = blocked;
    return output;
  }
  const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;

  // Do not seed the graph with unsettled odometry. A stable scan/pose window
  // also permits mapping when the robot remains stationary.
  if (config_.mapping_static_start_guard_enable) {
    if (!static_start_guard_configured_) {
      static_start_guard_ =
          StaticStartGuard(staticStartGuardOptionsFromConfig(config_));
      static_start_guard_configured_ = true;
    }
    if (!static_start_guard_.update(timestamp, *frame.undistorted_cloud,
                                   frame.T_world_lidar)) {
      auto output = makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
      output.mapping_block_reason = core::MappingBlockReason::StaticStartGuard;
      return output;
    }
  }

  auto &keyframes = session_->keyframeManager();
  if (!keyframes.shouldAddKeyframe(frame.T_world_lidar)) {
    Eigen::Isometry3d pose_map = frame.T_world_lidar;
    if (const auto latest = keyframes.getLatestKeyframe()) {
      // Non-keyframes use the latest optimized map<-odom anchor, without
      // writing that correction back into the raw odometry.
      pose_map = latest->pose_optimized * latest->pose_odom.inverse() *
                 frame.T_world_lidar;
    }
    if (!external_dense_trajectory_recording_enabled_) {
      appendDenseTrajectorySampleWithLatestAnchor(timestamp,
                                                  frame.T_world_lidar);
    }
    return makeOutput(true, pose_map, frame.undistorted_cloud);
  }

  const int64_t keyframe_id = keyframes.addKeyframe(
      timestamp, frame.T_world_lidar, frame.undistorted_cloud);
  session_->loopDetector().addDescriptor(keyframe_id, frame.undistorted_cloud);
  addRhpdDescriptorForKeyframe(keyframe_id, frame.undistorted_cloud);

  if (keyframe_id == 0) {
    session_->graphOptimizer().addPriorFactor(keyframe_id, frame.T_world_lidar);
  } else {
    addOdometryConstraint(keyframe_id, frame.T_world_lidar);
  }

  // The only absolute attitude the graph ever sees. Measured from this scan's
  // own returns, so it is independent of the pose it constrains.
  if (config_.floor_attitude_enable) {
    FloorNormalOptions floor_options;
    floor_options.max_radius_m = config_.floor_attitude_max_radius_m;
    floor_options.min_points = config_.floor_attitude_min_points;
    const auto floor = estimateFloorNormal(*frame.undistorted_cloud,
                                           frame.T_world_lidar, floor_options);
    if (floor.valid) {
      session_->graphOptimizer().addFloorAttitudeFactor(keyframe_id,
                                                        floor.normal_body);
      ++floor_attitude_accepted_;
    } else {
      ++floor_attitude_rejected_;
    }
  }

  if (config_.loop_geometry_enable)
    session_->graphOptimizer().appendPlaneObservations(
        keyframe_id, extractPlaneMoments(*frame.undistorted_cloud));
  session_->graphOptimizer().incrementalOptimize();
  refreshOptimizedPoses();

  if (session_->submapBuilder().enabled()) {
    const auto committed_keyframe = keyframes.getKeyframe(keyframe_id);
    if (!session_->submapBuilder().appendKeyframe(committed_keyframe)) {
      LOG(ERROR) << "Shadow submap append failed for committed keyframe id="
                << keyframe_id << '\n';
    } else {
      // graph_update necessarily runs before the just-committed keyframe can
      // join a shadow submap. Refresh once more so topology diagnostics see
      // complete ownership rather than remaining one keyframe behind.
      refreshSubmapPoses("submap_append");
    }
  }

  Eigen::Isometry3d optimized_pose = frame.T_world_lidar;
  if (session_->graphOptimizer().hasNode(keyframe_id)) {
    try {
      optimized_pose = session_->graphOptimizer().getOptimizedPose(keyframe_id);
    } catch (const std::exception &) {
      optimized_pose = frame.T_world_lidar;
    }
  }

  auto output = makeOutput(true, optimized_pose, frame.undistorted_cloud);
  output.accepted_keyframe = true;
  output.keyframe_id = keyframe_id;
  if (!external_dense_trajectory_recording_enabled_) {
    if (auto kf = keyframes.getKeyframe(keyframe_id)) {
      appendDenseTrajectorySample(timestamp, frame.T_world_lidar, keyframe_id,
                                  kf->pose_odom);
    }
  }

  {
    std::lock_guard<std::mutex> lock(loop_queue_mutex_);
    loop_detection_queue_.push_back(keyframe_id);
  }
  return output;
}

} // namespace n3mapping
