#include "n3mapping/humble/conversions.h"
#include "n3mapping/odometry_pose_validation.h"
#include "n3mapping/pcl_compat.h"

#include <pcl_conversions/pcl_conversions.h>

namespace n3mapping {

core::TimeStamp toCoreTimeStamp(const builtin_interfaces::msg::Time& stamp)
{
    core::TimeStamp out;
    out.nsec = static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
    return out;
}

Eigen::Isometry3d odometryPoseToIsometry(const nav_msgs::msg::Odometry& odom_msg)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    tryMakeRigidOdometryPose(
        odom_msg.pose.pose.position.x,
        odom_msg.pose.pose.position.y,
        odom_msg.pose.pose.position.z,
        odom_msg.pose.pose.orientation.x,
        odom_msg.pose.pose.orientation.y,
        odom_msg.pose.pose.orientation.z,
        odom_msg.pose.pose.orientation.w,
        &pose);
    return pose;
}

core::LioFrame toCoreLioFrame(const sensor_msgs::msg::PointCloud2& cloud_msg,
                              const nav_msgs::msg::Odometry& odom_msg)
{
    auto cloud = pcl::make_shared<core::LioFrame::PointCloud>();
    pcl::fromROSMsg(cloud_msg, *cloud);

    core::LioFrame frame;
    frame.stamp = toCoreTimeStamp(cloud_msg.header.stamp);
    frame.pose_valid = tryMakeRigidOdometryPose(
        odom_msg.pose.pose.position.x,
        odom_msg.pose.pose.position.y,
        odom_msg.pose.pose.position.z,
        odom_msg.pose.pose.orientation.x,
        odom_msg.pose.pose.orientation.y,
        odom_msg.pose.pose.orientation.z,
        odom_msg.pose.pose.orientation.w,
        &frame.T_world_lidar);
    frame.undistorted_cloud = cloud;
    return frame;
}

}  // namespace n3mapping
