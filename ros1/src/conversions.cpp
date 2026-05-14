#include "n3mapping_ros1/conversions.h"

#include <pcl_conversions/pcl_conversions.h>

namespace n3mapping {

core::TimeStamp toCoreTimeStamp(const ros::Time& stamp)
{
    core::TimeStamp out;
    out.nsec = static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nsec);
    return out;
}

Eigen::Isometry3d odometryPoseToIsometry(const nav_msgs::Odometry& odom_msg)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() << odom_msg.pose.pose.position.x,
                          odom_msg.pose.pose.position.y,
                          odom_msg.pose.pose.position.z;
    Eigen::Quaterniond q(odom_msg.pose.pose.orientation.w,
                         odom_msg.pose.pose.orientation.x,
                         odom_msg.pose.pose.orientation.y,
                         odom_msg.pose.pose.orientation.z);
    pose.linear() = q.normalized().toRotationMatrix();
    return pose;
}

core::LioFrame toCoreLioFrame(const sensor_msgs::PointCloud2& cloud_msg,
                              const nav_msgs::Odometry& odom_msg)
{
    auto cloud = pcl::make_shared<core::LioFrame::PointCloud>();
    pcl::fromROSMsg(cloud_msg, *cloud);

    core::LioFrame frame;
    frame.stamp = toCoreTimeStamp(cloud_msg.header.stamp);
    frame.T_world_lidar = odometryPoseToIsometry(odom_msg);
    frame.undistorted_cloud = cloud;
    frame.pose_valid = true;
    return frame;
}

}  // namespace n3mapping
