#include <limits>

#include <gtest/gtest.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "n3mapping/ros2/conversions.h"

namespace n3mapping {
namespace test {
namespace {

pcl::PointCloud<pcl::PointXYZI> makeCloud() {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::PointXYZI point;
    point.x = 1.0f;
    point.y = 2.0f;
    point.z = 3.0f;
    point.intensity = 4.0f;
    cloud.push_back(point);
    cloud.width = cloud.size();
    cloud.height = 1;
    cloud.is_dense = true;
    return cloud;
}

}  // namespace

TEST(Ros2ConversionsTest, ConvertsExternalLioMessages) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(makeCloud(), cloud_msg);
    cloud_msg.header.stamp.sec = 2;
    cloud_msg.header.stamp.nanosec = 300;

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.pose.pose.position.x = 1.0;
    odom_msg.pose.pose.position.y = 2.0;
    odom_msg.pose.pose.position.z = 3.0;
    odom_msg.pose.pose.orientation.w = 1.0;
    odom_msg.pose.covariance[0] = 0.5;

    const auto frame = ros2::externalLioFrameFromRos(cloud_msg, odom_msg);
    EXPECT_EQ(frame.stamp.nsec, 2000000300LL);
    EXPECT_NEAR(frame.T_world_lidar.translation().x(), 1.0, 1e-12);
    ASSERT_TRUE(frame.cloud);
    ASSERT_EQ(frame.cloud->size(), 1u);
    EXPECT_NEAR(frame.covariance(0, 0), 0.5, 1e-12);
    EXPECT_TRUE(frame.covariance_valid);
}

TEST(Ros2ConversionsTest, EmptyOrInvalidCovarianceFallsBackToIdentity) {
    nav_msgs::msg::Odometry odom_msg;
    EXPECT_TRUE(ros2::poseCovarianceFromOdom(odom_msg).isIdentity(1e-12));
    EXPECT_FALSE(ros2::poseCovarianceValid(odom_msg));

    odom_msg.pose.covariance[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(ros2::poseCovarianceFromOdom(odom_msg).isIdentity(1e-12));
    EXPECT_FALSE(ros2::poseCovarianceValid(odom_msg));
}

TEST(Ros2ConversionsTest, ConvertsRawLidarAndImuMessages) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(makeCloud(), cloud_msg);
    cloud_msg.header.stamp.sec = 3;
    cloud_msg.header.stamp.nanosec = 400;
    cloud_msg.header.frame_id = "lidar";

    const auto raw = ros2::rawLidarFrameFromRos(cloud_msg);
    EXPECT_EQ(raw.stamp_begin.nsec, 3000000400LL);
    EXPECT_EQ(raw.stamp_end.nsec, 3000000400LL);
    EXPECT_EQ(raw.frame_id, "lidar");
    EXPECT_EQ(raw.source_format, "pointcloud2");
    EXPECT_TRUE(raw.point_time_offsets_ns.empty());
    EXPECT_TRUE(raw.point_lines.empty());
    ASSERT_TRUE(raw.points);
    EXPECT_EQ(raw.points->size(), 1u);

    sensor_msgs::msg::Imu imu_msg;
    imu_msg.header.stamp.sec = 4;
    imu_msg.header.stamp.nanosec = 500;
    imu_msg.angular_velocity.z = 1.5;
    imu_msg.linear_acceleration.x = 9.8;
    imu_msg.orientation.w = 1.0;

    const auto imu = ros2::imuSampleFromRos(imu_msg);
    EXPECT_EQ(imu.stamp.nsec, 4000000500LL);
    EXPECT_NEAR(imu.angular_velocity.z(), 1.5, 1e-12);
    EXPECT_NEAR(imu.linear_accel.x(), 9.8, 1e-12);
    EXPECT_TRUE(imu.has_orientation);
}

TEST(Ros2ConversionsTest, PreservesPointCloud2PerPointTimingAndLine) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2Fields(
        6,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::msg::PointField::FLOAT32,
        "ring", 1, sensor_msgs::msg::PointField::UINT16,
        "time", 1, sensor_msgs::msg::PointField::FLOAT32);
    modifier.resize(2);
    cloud_msg.header.stamp.sec = 7;
    cloud_msg.header.frame_id = "velodyne";

    sensor_msgs::PointCloud2Iterator<float> x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> z(cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> intensity(cloud_msg, "intensity");
    sensor_msgs::PointCloud2Iterator<uint16_t> ring(cloud_msg, "ring");
    sensor_msgs::PointCloud2Iterator<float> time(cloud_msg, "time");

    *x = 1.0f;
    *y = 2.0f;
    *z = 3.0f;
    *intensity = 10.0f;
    *ring = 4u;
    *time = 0.001f;
    ++x;
    ++y;
    ++z;
    ++intensity;
    ++ring;
    ++time;
    *x = -1.0f;
    *y = -2.0f;
    *z = -3.0f;
    *intensity = 20.0f;
    *ring = 7u;
    *time = 0.0025f;

    const auto raw = ros2::rawLidarFrameFromRos(cloud_msg);
    EXPECT_EQ(raw.source_format, "pointcloud2");
    EXPECT_EQ(raw.stamp_begin.nsec, 7000000000LL);
    EXPECT_EQ(raw.stamp_end.nsec, 7002500000LL);
    ASSERT_TRUE(raw.points);
    ASSERT_EQ(raw.points->size(), 2u);
    EXPECT_NEAR(raw.points->at(1).intensity, 20.0f, 1e-6f);
    ASSERT_EQ(raw.point_lines.size(), 2u);
    ASSERT_EQ(raw.point_time_offsets_ns.size(), 2u);
    EXPECT_EQ(raw.point_lines[0], 4u);
    EXPECT_EQ(raw.point_lines[1], 7u);
    EXPECT_NEAR(static_cast<double>(raw.point_time_offsets_ns[0]), 1000000.0, 16.0);
    EXPECT_NEAR(static_cast<double>(raw.point_time_offsets_ns[1]), 2500000.0, 16.0);
}

#ifdef N3MAPPING_HAS_LIVOX_ROS_DRIVER2
TEST(Ros2ConversionsTest, ConvertsLivoxCustomMessage) {
    livox_ros_driver2::msg::CustomMsg msg;
    msg.header.frame_id = "livox";
    msg.header.stamp.sec = 10;
    msg.timebase = 123456789000ULL;
    msg.point_num = 2;

    livox_ros_driver2::msg::CustomPoint first;
    first.x = 1.0f;
    first.y = 2.0f;
    first.z = 3.0f;
    first.reflectivity = 42;
    first.offset_time = 100;
    first.line = 3;
    msg.points.push_back(first);

    livox_ros_driver2::msg::CustomPoint second;
    second.x = -1.0f;
    second.y = -2.0f;
    second.z = -3.0f;
    second.reflectivity = 7;
    second.offset_time = 250;
    second.line = 5;
    msg.points.push_back(second);

    const auto raw = ros2::rawLidarFrameFromLivoxCustom(msg);
    EXPECT_EQ(raw.stamp_begin.nsec, 123456789000LL);
    EXPECT_EQ(raw.stamp_end.nsec, 123456789250LL);
    EXPECT_EQ(raw.frame_id, "livox");
    EXPECT_EQ(raw.source_format, "livox_custom");
    ASSERT_TRUE(raw.points);
    ASSERT_EQ(raw.points->size(), 2u);
    EXPECT_NEAR(raw.points->at(0).intensity, 42.0f, 1e-6f);
    EXPECT_NEAR(raw.points->at(1).x, -1.0f, 1e-6f);
    ASSERT_EQ(raw.point_time_offsets_ns.size(), 2u);
    ASSERT_EQ(raw.point_lines.size(), 2u);
    EXPECT_EQ(raw.point_time_offsets_ns[0], 100u);
    EXPECT_EQ(raw.point_time_offsets_ns[1], 250u);
    EXPECT_EQ(raw.point_lines[0], 3u);
    EXPECT_EQ(raw.point_lines[1], 5u);
}
#endif

}  // namespace test
}  // namespace n3mapping
