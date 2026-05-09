#include <gtest/gtest.h>

#include "n3mapping/lio/dlio_deskew.h"

namespace n3mapping {
namespace test {
namespace {

core::RawLidarFrame makeFrame() {
    core::RawLidarFrame frame;
    frame.stamp_begin.nsec = 1000000000LL;
    frame.stamp_end.nsec = 1010000000LL;
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();
    for (int i = 0; i < 2; ++i) {
        pcl::PointXYZI point;
        point.x = 1.0f;
        point.y = static_cast<float>(i);
        point.z = 0.0f;
        point.intensity = static_cast<float>(i + 1);
        frame.points->push_back(point);
    }
    frame.points->width = static_cast<uint32_t>(frame.points->size());
    frame.points->height = 1;
    frame.points->is_dense = true;
    return frame;
}

lio::dlio::IntegratedPose makePose(double stamp, float tx) {
    lio::dlio::IntegratedPose pose;
    pose.stamp = stamp;
    pose.T = Eigen::Matrix4f::Identity();
    pose.T(0, 3) = tx;
    return pose;
}

}  // namespace

TEST(DlioDeskewTest, AppliesPerPointNearestTimedPose) {
    auto frame = makeFrame();
    frame.point_time_offsets_ns = {1000000u, 9000000u};
    const auto timing = lio::dlio::computeScanTiming(frame);
    const std::vector<lio::dlio::IntegratedPose,
                      Eigen::aligned_allocator<lio::dlio::IntegratedPose>>
        poses = {makePose(1.001, 10.0f), makePose(1.009, 20.0f)};

    const auto result = lio::dlio::deskewToWorld(frame, timing, poses);

    ASSERT_TRUE(result.valid);
    ASSERT_TRUE(result.cloud);
    ASSERT_EQ(result.cloud->size(), 2u);
    EXPECT_EQ(result.transformed_points, 2u);
    EXPECT_NEAR(result.cloud->at(0).x, 11.0f, 1e-6f);
    EXPECT_NEAR(result.cloud->at(1).x, 21.0f, 1e-6f);
    EXPECT_NEAR(result.cloud->at(1).intensity, 2.0f, 1e-6f);
}

TEST(DlioDeskewTest, FallsBackToMedianPoseWithoutPointTiming) {
    const auto frame = makeFrame();
    const auto timing = lio::dlio::computeScanTiming(frame);
    const std::vector<lio::dlio::IntegratedPose,
                      Eigen::aligned_allocator<lio::dlio::IntegratedPose>>
        poses = {makePose(1.005, 3.0f)};

    const auto result = lio::dlio::deskewToWorld(frame, timing, poses);

    ASSERT_TRUE(result.valid);
    ASSERT_EQ(result.cloud->size(), 2u);
    EXPECT_NEAR(result.cloud->at(0).x, 4.0f, 1e-6f);
    EXPECT_NEAR(result.cloud->at(1).x, 4.0f, 1e-6f);
}

TEST(DlioDeskewTest, CanDeskewIntoReferenceFrame) {
    auto frame = makeFrame();
    frame.point_time_offsets_ns = {1000000u, 9000000u};
    const auto timing = lio::dlio::computeScanTiming(frame);
    const std::vector<lio::dlio::IntegratedPose,
                      Eigen::aligned_allocator<lio::dlio::IntegratedPose>>
        poses = {makePose(1.001, 10.0f), makePose(1.009, 20.0f)};
    Eigen::Matrix4f T_world_reference = Eigen::Matrix4f::Identity();
    T_world_reference(0, 3) = 20.0f;

    const auto result = lio::dlio::deskewToReference(
        frame, timing, poses, T_world_reference);

    ASSERT_TRUE(result.valid);
    ASSERT_EQ(result.cloud->size(), 2u);
    EXPECT_NEAR(result.cloud->at(0).x, -9.0f, 1e-6f);
    EXPECT_NEAR(result.cloud->at(1).x, 1.0f, 1e-6f);
}

TEST(DlioDeskewTest, RejectsMissingInputs) {
    core::RawLidarFrame frame;
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();
    lio::dlio::ScanTiming timing;
    timing.valid = true;

    const auto result = lio::dlio::deskewToWorld(frame, timing, {});

    EXPECT_FALSE(result.valid);
    ASSERT_TRUE(result.cloud);
    EXPECT_TRUE(result.cloud->empty());
}

}  // namespace test
}  // namespace n3mapping
