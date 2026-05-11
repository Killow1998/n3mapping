#include <gtest/gtest.h>

#include "n3mapping/core/types.h"

namespace n3mapping {
namespace test {

TEST(CoreTypesTest, DefaultValuesAreRosFreeAndStable) {
    core::TimeStamp stamp;
    EXPECT_EQ(stamp.nsec, 0);

    core::ImuSample imu;
    EXPECT_EQ(imu.stamp.nsec, 0);
    EXPECT_TRUE(imu.linear_accel.isZero());
    EXPECT_TRUE(imu.angular_velocity.isZero());
    EXPECT_TRUE(imu.orientation.isApprox(Eigen::Quaterniond::Identity()));
    EXPECT_FALSE(imu.has_orientation);

    core::RawLidarFrame raw_frame;
    EXPECT_EQ(raw_frame.frame_id, "");
    EXPECT_EQ(raw_frame.source_format, "pointcloud2");
    EXPECT_EQ(raw_frame.points, nullptr);
    EXPECT_TRUE(raw_frame.point_time_offsets_ns.empty());
    EXPECT_TRUE(raw_frame.point_lines.empty());

    core::LioFrame lio_frame;
    EXPECT_EQ(lio_frame.stamp.nsec, 0);
    EXPECT_TRUE(lio_frame.T_world_lidar.isApprox(Eigen::Isometry3d::Identity()));
    EXPECT_EQ(lio_frame.undistorted_cloud, nullptr);
    EXPECT_TRUE(lio_frame.covariance.isApprox(Eigen::Matrix<double, 6, 6>::Identity()));
    EXPECT_FALSE(lio_frame.covariance_valid);
    EXPECT_FALSE(lio_frame.pose_valid);

    core::BackendOutput output;
    EXPECT_FALSE(output.success);
    EXPECT_FALSE(output.accepted_keyframe);
    EXPECT_FALSE(output.relocalization_locked);
    EXPECT_EQ(output.keyframe_id, -1);
    EXPECT_TRUE(output.T_world_lidar.isApprox(Eigen::Isometry3d::Identity()));
    EXPECT_EQ(output.cloud_body, nullptr);
    EXPECT_EQ(output.cloud_world, nullptr);
}

TEST(CoreTypesTest, PointCloudBackedFramesCanBePopulated) {
    auto cloud = std::make_shared<core::RawLidarFrame::PointCloud>();
    cloud->push_back(pcl::PointXYZI{1.0f, 2.0f, 3.0f, 4.0f});

    core::RawLidarFrame raw_frame;
    raw_frame.stamp_begin.nsec = 100;
    raw_frame.stamp_end.nsec = 200;
    raw_frame.frame_id = "lidar";
    raw_frame.source_format = "livox_custom";
    raw_frame.points = cloud;
    raw_frame.point_time_offsets_ns = {10U};
    raw_frame.point_lines = {3U};

    ASSERT_NE(raw_frame.points, nullptr);
    ASSERT_EQ(raw_frame.points->size(), 1U);
    EXPECT_EQ(raw_frame.frame_id, "lidar");
    EXPECT_EQ(raw_frame.source_format, "livox_custom");
    EXPECT_EQ(raw_frame.point_time_offsets_ns.front(), 10U);
    EXPECT_EQ(raw_frame.point_lines.front(), 3U);

    core::LioFrame lio_frame;
    lio_frame.stamp.nsec = 1234;
    lio_frame.pose_valid = true;
    lio_frame.covariance_valid = true;
    lio_frame.undistorted_cloud = cloud;
    lio_frame.T_world_lidar.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);

    ASSERT_NE(lio_frame.undistorted_cloud, nullptr);
    EXPECT_EQ(lio_frame.undistorted_cloud->size(), 1U);
    EXPECT_TRUE(lio_frame.pose_valid);
    EXPECT_TRUE(lio_frame.covariance_valid);
    EXPECT_DOUBLE_EQ(lio_frame.T_world_lidar.translation().x(), 1.0);
    EXPECT_DOUBLE_EQ(lio_frame.T_world_lidar.translation().y(), 2.0);
    EXPECT_DOUBLE_EQ(lio_frame.T_world_lidar.translation().z(), 3.0);
}

TEST(CoreTypesTest, BackendOutputCarriesCoreResults) {
    auto body_cloud = std::make_shared<core::LioFrame::PointCloud>();
    auto world_cloud = std::make_shared<core::LioFrame::PointCloud>();
    body_cloud->push_back(pcl::PointXYZI{0.0f, 0.0f, 0.0f, 1.0f});
    world_cloud->push_back(pcl::PointXYZI{5.0f, 0.0f, 0.0f, 1.0f});

    core::BackendOutput output;
    output.success = true;
    output.accepted_keyframe = true;
    output.relocalization_locked = true;
    output.keyframe_id = 42;
    output.cloud_body = body_cloud;
    output.cloud_world = world_cloud;
    output.T_world_lidar.translation() = Eigen::Vector3d(5.0, 6.0, 7.0);

    EXPECT_TRUE(output.success);
    EXPECT_TRUE(output.accepted_keyframe);
    EXPECT_TRUE(output.relocalization_locked);
    EXPECT_EQ(output.keyframe_id, 42);
    ASSERT_NE(output.cloud_body, nullptr);
    ASSERT_NE(output.cloud_world, nullptr);
    EXPECT_EQ(output.cloud_body->size(), 1U);
    EXPECT_EQ(output.cloud_world->size(), 1U);
    EXPECT_DOUBLE_EQ(output.T_world_lidar.translation().x(), 5.0);
    EXPECT_DOUBLE_EQ(output.T_world_lidar.translation().y(), 6.0);
    EXPECT_DOUBLE_EQ(output.T_world_lidar.translation().z(), 7.0);
}

}  // namespace test
}  // namespace n3mapping
