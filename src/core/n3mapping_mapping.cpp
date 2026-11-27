#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping_session.h"
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

  if (config_.odom_sanity_enable) {
    const auto& verdict = odometry_sanity_.check(timestamp, frame.T_world_lidar);
    if (verdict.diverged) {
      return core::MappingBlockReason::OdometryDiverged;
    }
    if (verdict.consecutive > 0) {
      return core::MappingBlockReason::InvalidInput;
    }
  }
  return core::MappingBlockReason::None;
}

core::BackendOutput
N3MappingCore::processMappingFrame(const core::LioFrame &frame) {
  extension_anchor_frame_.reset();
  const auto blocked = mappingInputBlockReason(frame);
  if (blocked != core::MappingBlockReason::None) {
    auto output = makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    output.mapping_block_reason = blocked;
    return output;
  }
  const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;

  if (config_.mapping_static_start_guard_enable) {
    if (!static_start_guard_.update(timestamp, *frame.undistorted_cloud,
                                   frame.T_world_lidar)) {
      auto output = makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
      output.mapping_block_reason = core::MappingBlockReason::StaticStartGuard;
      return output;
    }
  }

  auto &keyframes = session_->keyframes;
  if (!keyframes.shouldAddKeyframe(frame.T_world_lidar)) {
    Eigen::Isometry3d pose_map = frame.T_world_lidar;
    if (const auto latest = keyframes.getLatestKeyframe()) {

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
  session_->detector.addDescriptor(keyframe_id, frame.undistorted_cloud);
  addRhpdDescriptorForKeyframe(keyframe_id, frame.undistorted_cloud);

  if (keyframe_id == 0) {
    session_->optimizer.addPriorFactor(keyframe_id, frame.T_world_lidar);
  } else {
    addOdometryConstraint(keyframe_id, frame.T_world_lidar);
  }

  if (config_.floor_attitude_enable) {
    FloorNormalOptions floor_options;
    floor_options.max_radius_m = config_.floor_attitude_max_radius_m;
    floor_options.min_points = config_.floor_attitude_min_points;
    const auto floor = estimateFloorNormal(*frame.undistorted_cloud,
                                           frame.T_world_lidar, floor_options);
    if (floor.valid) {
      session_->optimizer.addFloorAttitudeFactor(keyframe_id,
                                                        floor.normal_body);
      ++floor_attitude_accepted_;
    } else {
      ++floor_attitude_rejected_;
    }
  }

  if (config_.loop_geometry_enable)
    session_->optimizer.appendPlaneObservations(
        keyframe_id, extractPlaneMoments(*frame.undistorted_cloud));
  session_->optimizer.incrementalOptimize();
  refreshOptimizedPoses();

  if (session_->submaps.enabled()) {
    const auto committed_keyframe = keyframes.getKeyframe(keyframe_id);
    if (!session_->submaps.appendKeyframe(committed_keyframe)) {
      LOG(ERROR) << "Shadow submap append failed for committed keyframe id="
                << keyframe_id << '\n';
    } else {

      refreshSubmapPoses("submap_append");
    }
  }

  Eigen::Isometry3d optimized_pose = frame.T_world_lidar;
  if (session_->optimizer.hasNode(keyframe_id)) {
    optimized_pose = session_->optimizer.getOptimizedPose(keyframe_id);
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
    loop_detection_queue_.push_back({keyframe_id});
  }
  return output;
}

}
