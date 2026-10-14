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

core::OdometrySource toCoreOdometrySource(const nav_msgs::msg::Odometry& odom_msg)
{
    core::OdometrySource source;
    source.stamp_nsec = toCoreTimeStamp(odom_msg.header.stamp).nsec;
    source.source_frame = odom_msg.header.frame_id;
    source.body_frame = odom_msg.child_frame_id;
    source.pose_valid = tryMakeRigidOdometryPose(
        odom_msg.pose.pose.position.x,
        odom_msg.pose.pose.position.y,
        odom_msg.pose.pose.position.z,
        odom_msg.pose.pose.orientation.x,
        odom_msg.pose.pose.orientation.y,
        odom_msg.pose.pose.orientation.z,
        odom_msg.pose.pose.orientation.w,
        &source.pose);
    return source;
}

Eigen::Isometry3d odometryPoseToIsometry(const nav_msgs::msg::Odometry& odom_msg)
{
    return toCoreOdometrySource(odom_msg).pose;
}

core::LioFrame toCoreLioFrame(const sensor_msgs::msg::PointCloud2& cloud_msg,
                              const nav_msgs::msg::Odometry& odom_msg)
{
    auto cloud = pcl::make_shared<core::LioFrame::PointCloud>();
    pcl::fromROSMsg(cloud_msg, *cloud);

    const auto source = toCoreOdometrySource(odom_msg);
    core::LioFrame frame;
    frame.stamp = toCoreTimeStamp(cloud_msg.header.stamp);
    frame.source_frame_id = source.source_frame;
    frame.pose_valid = source.pose_valid;
    frame.T_world_lidar = source.pose;
    frame.undistorted_cloud = cloud;
    return frame;
}

}  // namespace n3mapping
