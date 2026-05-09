#include <gtest/gtest.h>

#include "n3mapping/lio/dlio_core.h"

namespace n3mapping {
namespace test {
namespace {

core::RawLidarFrame makeFrame() {
    core::RawLidarFrame frame;
    frame.source_format = "pointcloud2";
    frame.stamp_begin.nsec = 3000000000LL;
    frame.stamp_end.nsec = 3001000000LL;
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();
    pcl::PointXYZI point;
    point.x = 3.0f;
    point.y = 0.0f;
    point.z = 0.0f;
    point.intensity = 6.0f;
    frame.points->push_back(point);
    frame.point_time_offsets_ns.push_back(1000000u);
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

TEST(DlioCoreTest, AcceptsImuAndLidarAtCoreBoundary) {
    lio::LioFrontendConfig config;
    config.dlio_time_encoding = "velodyne";
    lio::dlio::Core core(config);

    core.addImu(makeImu(3000000000LL));
    core.addImu(makeImu(3001000000LL));
    const auto output = core.addLidar(makeFrame());

    EXPECT_FALSE(core.implemented());
    EXPECT_FALSE(output.has_value());
    EXPECT_EQ(core.imuSamplesSeen(), 2u);
    EXPECT_EQ(core.lidarFramesSeen(), 1u);
    EXPECT_EQ(core.lastInputPacket().cloud_stats.input_points, 1u);
    EXPECT_EQ(core.lastInputPacket().cloud_stats.output_points, 1u);
    EXPECT_EQ(core.lastInputPacket().time_encoding,
              lio::dlio::TimeEncoding::VelodyneOffsetSeconds);
    EXPECT_EQ(core.lastInputPacket().imu_samples.size(), 2u);
    EXPECT_TRUE(core.lastInputPacket().has_complete_imu_window);
    ASSERT_TRUE(core.lastImuPropagation().has_value());
    EXPECT_TRUE(core.lastImuPropagation()->valid);
    EXPECT_NEAR(core.lastImuPropagation()->velocity.x(), 0.001, 1e-12);
    ASSERT_TRUE(core.predictedState().has_value());
    EXPECT_TRUE(core.predictedState()->initialized);
    EXPECT_NEAR(core.predictedState()->velocity_world.x(), 0.001, 1e-12);
}

TEST(DlioCoreTest, CanReturnPredictionOnlyFrameWhenEnabled) {
    lio::LioFrontendConfig config;
    config.prediction_only_output = true;
    lio::dlio::Core core(config);

    core.addImu(makeImu(3000000000LL));
    core.addImu(makeImu(3001000000LL));
    const auto output = core.addLidar(makeFrame());

    ASSERT_TRUE(output.has_value());
    EXPECT_TRUE(output->pose_valid);
    ASSERT_TRUE(output->undistorted_cloud);
    EXPECT_EQ(output->undistorted_cloud->size(), 1u);
    EXPECT_NEAR(output->T_world_lidar.translation().x(), 0.0000005, 1e-12);
    ASSERT_TRUE(core.localMapCloud());
    EXPECT_EQ(core.localMapCloud()->size(), 1u);
    ASSERT_TRUE(core.denseMapCloud());
    EXPECT_EQ(core.denseMapCloud()->size(), 1u);
    EXPECT_TRUE(core.lastDenseMapAddResult().accepted);
    EXPECT_FALSE(core.lastAlignmentStats().valid);
}

TEST(DlioCoreTest, RespectsDenseMapInputSkip) {
    lio::LioFrontendConfig config;
    config.prediction_only_output = true;
    config.dlio_dense_input_skip = 2;
    lio::dlio::Core core(config);

    core.addImu(makeImu(3000000000LL));
    core.addImu(makeImu(3001000000LL));
    ASSERT_TRUE(core.addLidar(makeFrame()).has_value());
    EXPECT_FALSE(core.lastDenseMapAddResult().accepted);
    ASSERT_TRUE(core.denseMapCloud());
    EXPECT_TRUE(core.denseMapCloud()->empty());

    core::RawLidarFrame second = makeFrame();
    second.stamp_begin.nsec = 3001000000LL;
    second.stamp_end.nsec = 3002000000LL;
    core.addImu(makeImu(3002000000LL));
    ASSERT_TRUE(core.addLidar(second).has_value());
    EXPECT_TRUE(core.lastDenseMapAddResult().accepted);
    EXPECT_EQ(core.denseMapCloud()->size(), 1u);
}

TEST(DlioCoreTest, ResetClearsBufferedBoundaryState) {
    lio::dlio::Core core;
    core.addImu(makeImu(1));
    core.addLidar(makeFrame());
    core.reset();

    EXPECT_EQ(core.imuSamplesSeen(), 0u);
    EXPECT_EQ(core.lidarFramesSeen(), 0u);
    EXPECT_FALSE(core.lastInputPacket().cloud);
    EXPECT_TRUE(core.lastInputPacket().imu_samples.empty());
    EXPECT_FALSE(core.lastImuPropagation().has_value());
    EXPECT_FALSE(core.predictedState().has_value());
    ASSERT_TRUE(core.localMapCloud());
    EXPECT_TRUE(core.localMapCloud()->empty());
    ASSERT_TRUE(core.denseMapCloud());
    EXPECT_TRUE(core.denseMapCloud()->empty());
    EXPECT_FALSE(core.lastDenseMapAddResult().accepted);
    EXPECT_FALSE(core.lastAlignmentStats().valid);
}

TEST(DlioCoreTest, AppliesLocalMapCentroidCorrection) {
    lio::LioFrontendConfig config;
    config.prediction_only_output = true;
    lio::dlio::Core core(config);

    core.addImu(makeImu(3000000000LL));
    core.addImu(makeImu(3001000000LL));
    ASSERT_TRUE(core.addLidar(makeFrame()).has_value());

    core::RawLidarFrame second = makeFrame();
    second.points->at(0).x += 0.25f;
    second.stamp_begin.nsec = 3001000000LL;
    second.stamp_end.nsec = 3002000000LL;
    core.addImu(makeImu(3002000000LL));
    const auto corrected = core.addLidar(second);

    ASSERT_TRUE(corrected.has_value());
    EXPECT_TRUE(core.lastAlignmentStats().valid);
    EXPECT_NEAR(core.lastAlignmentStats().centroid_correction_world.x(),
                -0.2500015, 1e-6);
    EXPECT_NEAR(corrected->T_world_lidar.translation().x(), -0.249999, 1e-6);
}

TEST(DlioCoreTest, RespectsAlignmentCorrespondenceGate) {
    lio::LioFrontendConfig config;
    config.prediction_only_output = true;
    config.alignment_max_correspondence_distance = 0.05;
    lio::dlio::Core core(config);

    core.addImu(makeImu(3000000000LL));
    core.addImu(makeImu(3001000000LL));
    ASSERT_TRUE(core.addLidar(makeFrame()).has_value());

    core::RawLidarFrame second = makeFrame();
    second.points->at(0).x += 0.25f;
    second.stamp_begin.nsec = 3001000000LL;
    second.stamp_end.nsec = 3002000000LL;
    core.addImu(makeImu(3002000000LL));
    const auto uncorrected = core.addLidar(second);

    ASSERT_TRUE(uncorrected.has_value());
    EXPECT_FALSE(core.lastAlignmentStats().valid);
    EXPECT_NEAR(uncorrected->T_world_lidar.translation().x(), 0.000002, 1e-12);
}

TEST(DlioCoreTest, CarriesPredictionAcrossLidarFrames) {
    lio::dlio::Core core;
    core.addImu(makeImu(3000000000LL));
    core.addImu(makeImu(3001000000LL));
    core.addLidar(makeFrame());
    ASSERT_TRUE(core.predictedState().has_value());
    const double first_velocity = core.predictedState()->velocity_world.x();

    core::RawLidarFrame second = makeFrame();
    second.stamp_begin.nsec = 3001000000LL;
    second.stamp_end.nsec = 3002000000LL;
    core.addImu(makeImu(3002000000LL));
    core.addLidar(second);

    ASSERT_TRUE(core.predictedState().has_value());
    EXPECT_GT(core.predictedState()->velocity_world.x(), first_velocity);
    EXPECT_NEAR(core.predictedState()->velocity_world.x(), 0.002, 1e-12);
}

TEST(DlioCoreTest, ReportsExtractionStatus) {
    lio::dlio::Core core;
    EXPECT_EQ(core.capability(), lio::FrontendCapability::PredictionOnly);
    EXPECT_FALSE(core.implemented());
    EXPECT_NE(std::string(lio::dlio::coreStatus()).find("input boundary"),
              std::string::npos);
}

}  // namespace test
}  // namespace n3mapping
