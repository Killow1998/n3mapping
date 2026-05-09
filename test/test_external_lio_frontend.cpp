#include <limits>

#include <gtest/gtest.h>

#include "n3mapping/lio/external_frontend.h"

namespace n3mapping {
namespace test {
namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr makeCloud() {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PointXYZI point;
    point.x = 1.0f;
    point.y = 2.0f;
    point.z = 3.0f;
    point.intensity = 4.0f;
    cloud->push_back(point);
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

}  // namespace

TEST(ExternalLioFrontendTest, ConvertsSynchronizedExternalFrame) {
    lio::ExternalLioFrontend frontend;

    core::TimeStamp stamp;
    stamp.nsec = 123456789;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() << 1.0, 2.0, 3.0;
    Eigen::Matrix<double, 6, 6> covariance =
        Eigen::Matrix<double, 6, 6>::Identity() * 0.25;

    auto output = frontend.addSynchronizedFrame(stamp, pose, makeCloud(), covariance);
    ASSERT_TRUE(output.has_value());
    EXPECT_TRUE(output->pose_valid);
    EXPECT_EQ(output->stamp.nsec, stamp.nsec);
    EXPECT_NEAR(output->T_world_lidar.translation().x(), 1.0, 1e-9);
    EXPECT_EQ(output->undistorted_cloud->size(), 1u);
    EXPECT_NEAR(output->covariance(0, 0), 0.25, 1e-12);
}

TEST(ExternalLioFrontendTest, RejectsInvalidExternalFrame) {
    lio::ExternalLioFrontend frontend;

    core::TimeStamp stamp;
    stamp.nsec = 1;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    EXPECT_FALSE(frontend.addSynchronizedFrame(stamp, pose, nullptr).has_value());

    pose.matrix()(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(frontend.addSynchronizedFrame(stamp, pose, makeCloud()).has_value());
}

TEST(ExternalLioFrontendTest, RawLidarUsesLatestExternalPose) {
    lio::ExternalLioFrontend frontend;
    core::RawLidarFrame raw;
    raw.stamp_begin.nsec = 10;
    raw.stamp_end.nsec = 20;
    raw.points = makeCloud();
    EXPECT_FALSE(frontend.addLidar(raw).has_value());

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation().x() = 5.0;
    ASSERT_TRUE(frontend.addSynchronizedFrame(core::TimeStamp{5}, pose, makeCloud()).has_value());

    auto output = frontend.addLidar(raw);
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output->stamp.nsec, 20);
    EXPECT_NEAR(output->T_world_lidar.translation().x(), 5.0, 1e-9);
    EXPECT_EQ(output->undistorted_cloud->size(), 1u);
}

}  // namespace test
}  // namespace n3mapping
