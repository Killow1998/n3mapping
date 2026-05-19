#pragma once

#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>

#include "n3mapping/core/types.h"

namespace n3mapping {

core::TimeStamp toCoreTimeStamp(const ros::Time& stamp);
Eigen::Isometry3d odometryPoseToIsometry(const nav_msgs::Odometry& odom_msg);
core::LioFrame toCoreLioFrame(const sensor_msgs::PointCloud2& cloud_msg,
                              const nav_msgs::Odometry& odom_msg);

}  // namespace n3mapping
