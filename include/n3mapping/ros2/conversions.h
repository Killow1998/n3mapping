// ROS 2 message conversion helpers for the n3mapping core data contracts.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#ifdef N3MAPPING_HAS_LIVOX_ROS_DRIVER2
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif

#include "n3mapping/core/types.h"

namespace n3mapping {
namespace ros2 {

struct ExternalLioRosFrame {
    core::TimeStamp stamp;
    Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    Eigen::Matrix<double, 6, 6> covariance =
        Eigen::Matrix<double, 6, 6>::Identity();
};

core::TimeStamp toCoreStamp(const builtin_interfaces::msg::Time& stamp);
Eigen::Isometry3d poseFromOdom(const nav_msgs::msg::Odometry& odom_msg);
Eigen::Matrix<double, 6, 6> poseCovarianceFromOdom(
    const nav_msgs::msg::Odometry& odom_msg);
ExternalLioRosFrame externalLioFrameFromRos(
    const sensor_msgs::msg::PointCloud2& cloud_msg,
    const nav_msgs::msg::Odometry& odom_msg);
core::ImuSample imuSampleFromRos(const sensor_msgs::msg::Imu& imu_msg);
core::RawLidarFrame rawLidarFrameFromRos(
    const sensor_msgs::msg::PointCloud2& cloud_msg);
#ifdef N3MAPPING_HAS_LIVOX_ROS_DRIVER2
core::RawLidarFrame rawLidarFrameFromLivoxCustom(
    const livox_ros_driver2::msg::CustomMsg& msg);
#endif

}  // namespace ros2
}  // namespace n3mapping
