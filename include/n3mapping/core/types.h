// ROS-free core data contracts for n3mapping.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/relocalization_state.h"

namespace n3mapping {
namespace core {

struct TimeStamp {
  int64_t nsec = 0;
};

struct ImuSample {
  TimeStamp stamp;
  Eigen::Vector3d linear_accel = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  bool has_orientation = false;
};

struct RawLidarFrame {
  using PointCloud = pcl::PointCloud<pcl::PointXYZI>;

  TimeStamp stamp_begin;
  TimeStamp stamp_end;
  std::string frame_id;
  std::string source_format = "pointcloud2";
  std::vector<uint32_t> point_time_offsets_ns;
  std::vector<uint8_t> point_lines;
  PointCloud::Ptr points;
};

struct LioFrame {
  using PointCloud = pcl::PointCloud<pcl::PointXYZI>;

  TimeStamp stamp;
  // Frame in which T_world_lidar is expressed by the upstream LIO source.
  std::string source_frame_id;
  Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();
  PointCloud::Ptr undistorted_cloud;
  Eigen::Matrix<double, 6, 6> covariance =
      Eigen::Matrix<double, 6, 6>::Identity();
  bool covariance_valid = false;
  bool pose_valid = false;
};

struct BackendOutput {
  bool success = false;
  bool accepted_keyframe = false;
  bool relocalization_locked = false;
  RelocalizationState relocalization_state = RelocalizationState::SEARCHING;
  PoseSource pose_source = PoseSource::NONE;
  // Decision from the relocalization attempt made for this frame. "tracking"
  // means the already-locked tracker supplied the pose without a new attempt.
  std::string relocalization_decision = "not_attempted";
  int64_t keyframe_id = -1;
  int64_t relocalization_seed_keyframe_id = -1;
  int64_t relocalization_support_keyframe_id = -1;
  // Legacy alias for relocalization_support_keyframe_id.
  int64_t matched_keyframe_id = -1;
  Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();
  LioFrame::PointCloud::Ptr cloud_body;
  LioFrame::PointCloud::Ptr cloud_world;
};

struct DenseTrajectoryPose {
  uint64_t seq = 0;
  double timestamp = 0.0;
  Eigen::Isometry3d pose_world_lidar = Eigen::Isometry3d::Identity();
};

struct DenseTrajectoryMetadata {
  std::string source = "none";
  bool degraded = true;
};

struct AnchoredDenseTrajectorySample {
  uint64_t seq = 0;
  double timestamp = 0.0;
  Eigen::Isometry3d pose_world_lidar_raw = Eigen::Isometry3d::Identity();
  int64_t anchor_keyframe_id = -1;
  Eigen::Isometry3d anchor_pose_world_lidar_raw = Eigen::Isometry3d::Identity();
  bool has_anchor = false;
  bool use_bracketing_correction = true;
};

} // namespace core
} // namespace n3mapping
