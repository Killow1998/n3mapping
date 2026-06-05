// ROS-free core data contracts for n3mapping.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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
    int64_t keyframe_id = -1;
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

}  // namespace core
}  // namespace n3mapping
