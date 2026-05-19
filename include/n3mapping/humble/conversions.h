#pragma once

#include <Eigen/Geometry>
#include <builtin_interfaces/msg/time.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "n3mapping/core/types.h"

namespace n3mapping {

core::TimeStamp toCoreTimeStamp(const builtin_interfaces::msg::Time& stamp);
Eigen::Isometry3d odometryPoseToIsometry(const nav_msgs::msg::Odometry& odom_msg);
core::LioFrame toCoreLioFrame(const sensor_msgs::msg::PointCloud2& cloud_msg,
                              const nav_msgs::msg::Odometry& odom_msg);

}  // namespace n3mapping
