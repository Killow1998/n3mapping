#include <gtest/gtest.h>
#include <pcl_conversions/pcl_conversions.h>

#include "n3mapping/humble/conversions.h"

namespace n3mapping {
namespace test {

TEST(HumbleConversionsTest, TimestampUsesNanoseconds)
{
    builtin_interfaces::msg::Time stamp;
    stamp.sec = 12;
    stamp.nanosec = 345;

    const auto core_stamp = toCoreTimeStamp(stamp);

    EXPECT_EQ(core_stamp.nsec, 12000000345LL);
}

TEST(HumbleConversionsTest, OdometryPoseConvertsToIsometry)
{
    nav_msgs::msg::Odometry odom;
    odom.pose.pose.position.x = 1.0;
    odom.pose.pose.position.y = 2.0;
    odom.pose.pose.position.z = 3.0;
    const Eigen::Quaterniond q(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()));
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();

    const auto pose = odometryPoseToIsometry(odom);

    EXPECT_DOUBLE_EQ(pose.translation().x(), 1.0);
    EXPECT_DOUBLE_EQ(pose.translation().y(), 2.0);
    EXPECT_DOUBLE_EQ(pose.translation().z(), 3.0);
    EXPECT_TRUE(pose.linear().isApprox(q.toRotationMatrix()));
}

TEST(HumbleConversionsTest, PointCloudAndOdometryBuildCoreLioFrame)
{
    pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
    pcl_cloud.push_back(pcl::PointXYZI{1.0f, 2.0f, 3.0f, 4.0f});
    pcl_cloud.width = 1;
    pcl_cloud.height = 1;
    pcl_cloud.is_dense = true;

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(pcl_cloud, cloud_msg);
    cloud_msg.header.stamp.sec = 2;
    cloud_msg.header.stamp.nanosec = 50;
    cloud_msg.header.frame_id = "lidar";

    nav_msgs::msg::Odometry odom;
    odom.pose.pose.position.x = 5.0;
    odom.pose.pose.orientation.w = 1.0;

    const auto frame = toCoreLioFrame(cloud_msg, odom);

    EXPECT_TRUE(frame.pose_valid);
    EXPECT_EQ(frame.stamp.nsec, 2000000050LL);
    ASSERT_NE(frame.undistorted_cloud, nullptr);
    ASSERT_EQ(frame.undistorted_cloud->size(), 1U);
    EXPECT_FLOAT_EQ(frame.undistorted_cloud->front().x, 1.0f);
    EXPECT_DOUBLE_EQ(frame.T_world_lidar.translation().x(), 5.0);
}

}  // namespace test
}  // namespace n3mapping
