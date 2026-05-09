#include <gtest/gtest.h>

#include "n3mapping/lio/fast_lio_core.h"

namespace n3mapping {
namespace test {
namespace {

core::RawLidarFrame makeFrame() {
    core::RawLidarFrame frame;
    frame.stamp_begin.nsec = 4000000000LL;
    frame.stamp_end.nsec = 4001000000LL;
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();
    pcl::PointXYZI point;
    point.x = 4.0f;
    point.y = 0.0f;
    point.z = 0.0f;
    point.intensity = 5.0f;
    frame.points->push_back(point);
    frame.point_time_offsets_ns.push_back(2000000u);
    frame.point_lines.push_back(2u);
    frame.points->width = 1;
    frame.points->height = 1;
    frame.points->is_dense = true;
    return frame;
}

core::ImuSample makeImu(int64_t stamp_nsec) {
    core::ImuSample sample;
    sample.stamp.nsec = stamp_nsec;
    sample.linear_accel.x() = 1.0;
    return sample;
}

}  // namespace

TEST(FastLioCoreTest, AcceptsImuAndLidarAtCoreBoundary) {
    lio::LioFrontendConfig config;
    config.scan_lines = 16;
    lio::fast_lio::Core core(config);

    core.addImu(makeImu(4000000000LL));
    core.addImu(makeImu(4001000000LL));
    const auto output = core.addLidar(makeFrame());

    EXPECT_FALSE(core.implemented());
    EXPECT_FALSE(output.has_value());
    EXPECT_EQ(core.imuSamplesSeen(), 2u);
    EXPECT_EQ(core.lidarFramesSeen(), 1u);
    EXPECT_EQ(core.lastInputPacket().cloud_stats.input_points, 1u);
    EXPECT_EQ(core.lastInputPacket().cloud_stats.output_points, 1u);
    ASSERT_TRUE(core.lastInputPacket().cloud);
    ASSERT_EQ(core.lastInputPacket().cloud->size(), 1u);
    EXPECT_NEAR(core.lastInputPacket().cloud->at(0).curvature, 2.0f, 1e-6f);
    EXPECT_EQ(core.lastInputPacket().imu_samples.size(), 2u);
    EXPECT_TRUE(core.lastInputPacket().has_complete_imu_window);
    ASSERT_TRUE(core.lastImuPropagation().has_value());
    EXPECT_TRUE(core.lastImuPropagation()->valid);
    EXPECT_NEAR(core.lastImuPropagation()->velocity.x(), 0.001, 1e-12);
    ASSERT_TRUE(core.predictedState().has_value());
    EXPECT_TRUE(core.predictedState()->initialized);
    EXPECT_NEAR(core.predictedState()->velocity_world.x(), 0.001, 1e-12);
}

TEST(FastLioCoreTest, ResetClearsBufferedBoundaryState) {
    lio::fast_lio::Core core;
    core.addImu(makeImu(1));
    core.addLidar(makeFrame());
    core.reset();

    EXPECT_EQ(core.imuSamplesSeen(), 0u);
    EXPECT_EQ(core.lidarFramesSeen(), 0u);
    EXPECT_FALSE(core.lastInputPacket().cloud);
    EXPECT_TRUE(core.lastInputPacket().imu_samples.empty());
    EXPECT_FALSE(core.lastImuPropagation().has_value());
    EXPECT_FALSE(core.predictedState().has_value());
}

TEST(FastLioCoreTest, ReportsExtractionStatus) {
    EXPECT_NE(std::string(lio::fast_lio::coreStatus()).find("input boundary"),
              std::string::npos);
}

}  // namespace test
}  // namespace n3mapping
