#include <cmath>

#include <gtest/gtest.h>

#ifdef N3MAPPING_BUILD_DLIO_CORE
#include "n3mapping/lio/dlio_core.h"
#include "n3mapping/lio/dlio_frontend.h"
#endif
#include "n3mapping/lio/external_frontend.h"
#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
#include "n3mapping/lio/fast_lio_core.h"
#include "n3mapping/lio/fast_lio_frontend.h"
#endif
#include "n3mapping/lio/frontend_config.h"
#include "n3mapping/lio/frontend_factory.h"

namespace n3mapping {
namespace test {
namespace {

#if defined(N3MAPPING_BUILD_FAST_LIO_CORE) || defined(N3MAPPING_BUILD_DLIO_CORE)
core::RawLidarFrame makeRawLidarFrame(const std::string& source_format = "pointcloud2") {
    core::RawLidarFrame frame;
    frame.source_format = source_format;
    frame.stamp_begin.nsec = 1000000000LL;
    frame.stamp_end.nsec = 1001000000LL;
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();
    pcl::PointXYZI point;
    point.x = 1.0f;
    point.y = 2.0f;
    point.z = 3.0f;
    point.intensity = 4.0f;
    frame.points->push_back(point);
    frame.point_time_offsets_ns.push_back(1000000u);
    frame.points->width = static_cast<uint32_t>(frame.points->size());
    frame.points->height = 1;
    return frame;
}

core::ImuSample makeImuSample(int64_t stamp_nsec) {
    core::ImuSample sample;
    sample.stamp.nsec = stamp_nsec;
    return sample;
}
#endif

}  // namespace

TEST(LioFrontendFactoryTest, CreatesExternalFrontendByDefault) {
    Config config;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_EQ(result.mode, lio::FrontendMode::External);
    EXPECT_NE(dynamic_cast<lio::ExternalLioFrontend*>(result.frontend.get()), nullptr);
    EXPECT_EQ(result.frontend->capability(), lio::FrontendCapability::ExternalFrameAdapter);
}

TEST(LioFrontendFactoryTest, ParsesAliases) {
    EXPECT_EQ(lio::parseFrontendMode("external"), lio::FrontendMode::External);
    EXPECT_EQ(lio::parseFrontendMode("FAST_LIO2"), lio::FrontendMode::FastLio);
    EXPECT_EQ(lio::parseFrontendMode("fastlio"), lio::FrontendMode::FastLio);
    EXPECT_EQ(lio::parseFrontendMode("DLIO"), lio::FrontendMode::Dlio);
    EXPECT_EQ(lio::parseFrontendMode("bad"), lio::FrontendMode::Unknown);
}

TEST(LioFrontendFactoryTest, NamesFrontendCapabilities) {
    EXPECT_STREQ(lio::frontendCapabilityName(lio::FrontendCapability::Unavailable),
                 "unavailable");
    EXPECT_STREQ(lio::frontendCapabilityName(lio::FrontendCapability::ExternalFrameAdapter),
                 "external_frame_adapter");
    EXPECT_STREQ(lio::frontendCapabilityName(lio::FrontendCapability::PredictionOnly),
                 "prediction_only");
    EXPECT_STREQ(lio::frontendCapabilityName(lio::FrontendCapability::FullOdometry),
                 "full_odometry");
}

TEST(LioFrontendFactoryTest, BuiltinFrontendsAreExplicitlyUnsupportedForNow) {
    Config config;
    config.frontend_mode = "fast_lio";
    auto fast_lio = lio::createLioFrontend(config);
#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
    EXPECT_TRUE(fast_lio.ok()) << fast_lio.error;
    ASSERT_TRUE(fast_lio.frontend);
    EXPECT_EQ(fast_lio.frontend->capability(), lio::FrontendCapability::PredictionOnly);
#else
    EXPECT_FALSE(fast_lio.ok());
    EXPECT_NE(fast_lio.error.find("fast_lio_core"), std::string::npos);
#endif
    EXPECT_EQ(fast_lio.mode, lio::FrontendMode::FastLio);

    config.frontend_mode = "dlio";
    auto dlio = lio::createLioFrontend(config);
#ifdef N3MAPPING_BUILD_DLIO_CORE
    EXPECT_TRUE(dlio.ok()) << dlio.error;
    ASSERT_TRUE(dlio.frontend);
    EXPECT_EQ(dlio.frontend->capability(), lio::FrontendCapability::PredictionOnly);
#else
    EXPECT_FALSE(dlio.ok());
    EXPECT_NE(dlio.error.find("dlio_core"), std::string::npos);
#endif
    EXPECT_EQ(dlio.mode, lio::FrontendMode::Dlio);
}

TEST(LioFrontendFactoryTest, DerivesFrontendConfigFromCoreConfig) {
    Config config;
    config.lidar_type = "mid360";
    config.frontend_time_offset = 0.012;
    config.frontend_publish_debug = true;
    config.frontend_debug_publish_odom = true;
    config.frontend_debug_publish_deskewed_cloud = true;
    config.frontend_debug_publish_local_map = false;
    config.frontend_debug_publish_timing = true;
    config.frontend_imu_buffer_max_samples = 17;
    config.frontend_point_filter_num = 3;
    config.frontend_scan_lines = 32;
    config.frontend_blind = 0.25;
    config.frontend_max_abs_coordinate = 99.0;
    config.frontend_alignment_max_correspondence_distance = 0.75;
    config.frontend_prediction_only_output = true;
    config.dlio_time_encoding = "velodyne";
    config.dlio_dense_map_leaf_size = 0.25;
    config.dlio_dense_input_skip = 4;
    config.frontend_lidar_to_body_tx = 1.0;
    config.frontend_lidar_to_body_ty = 2.0;
    config.frontend_lidar_to_body_tz = 3.0;
    config.frontend_lidar_to_body_yaw = 0.5;
    config.frontend_imu_to_body_tx = -1.0;
    config.frontend_imu_to_body_pitch = 0.25;

    const auto frontend_config = lio::makeLioFrontendConfig(config);
    EXPECT_EQ(frontend_config.lidar_type, "mid360");
    EXPECT_NEAR(frontend_config.time_offset, 0.012, 1e-12);
    EXPECT_TRUE(frontend_config.publish_debug);
    EXPECT_TRUE(frontend_config.debug_publish_odom);
    EXPECT_TRUE(frontend_config.debug_publish_deskewed_cloud);
    EXPECT_FALSE(frontend_config.debug_publish_local_map);
    EXPECT_TRUE(frontend_config.debug_publish_timing);
    EXPECT_EQ(frontend_config.imu_buffer_max_samples, 17u);
    EXPECT_EQ(frontend_config.point_filter_num, 3u);
    EXPECT_EQ(frontend_config.scan_lines, 32u);
    EXPECT_NEAR(frontend_config.blind, 0.25, 1e-12);
    EXPECT_NEAR(frontend_config.max_abs_coordinate, 99.0, 1e-12);
    EXPECT_NEAR(frontend_config.alignment_max_correspondence_distance, 0.75, 1e-12);
    EXPECT_TRUE(frontend_config.prediction_only_output);
    EXPECT_EQ(frontend_config.dlio_time_encoding, "velodyne");
    EXPECT_NEAR(frontend_config.dlio_dense_map_leaf_size, 0.25, 1e-12);
    EXPECT_EQ(frontend_config.dlio_dense_input_skip, 4u);
    EXPECT_NEAR(frontend_config.T_body_lidar.translation().x(), 1.0, 1e-12);
    EXPECT_NEAR(frontend_config.T_body_lidar.translation().y(), 2.0, 1e-12);
    EXPECT_NEAR(frontend_config.T_body_lidar.translation().z(), 3.0, 1e-12);
    EXPECT_NEAR(frontend_config.T_body_lidar.linear()(0, 0), std::cos(0.5), 1e-12);
    EXPECT_NEAR(frontend_config.T_body_lidar.linear()(1, 0), std::sin(0.5), 1e-12);
    EXPECT_NEAR(frontend_config.T_body_imu.translation().x(), -1.0, 1e-12);
    EXPECT_NEAR(frontend_config.T_body_imu.linear()(0, 2), std::sin(0.25), 1e-12);
}

TEST(LioFrontendFactoryTest, DebugOutputsRequireMasterSwitch) {
    Config config;
    config.frontend_publish_debug = false;
    config.frontend_debug_publish_odom = true;
    config.frontend_debug_publish_deskewed_cloud = true;
    config.frontend_debug_publish_local_map = true;
    config.frontend_debug_publish_timing = true;

    const auto frontend_config = lio::makeLioFrontendConfig(config);
    EXPECT_FALSE(frontend_config.publish_debug);
    EXPECT_FALSE(frontend_config.debug_publish_odom);
    EXPECT_FALSE(frontend_config.debug_publish_deskewed_cloud);
    EXPECT_FALSE(frontend_config.debug_publish_local_map);
    EXPECT_FALSE(frontend_config.debug_publish_timing);
}

#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
TEST(LioFrontendFactoryTest, FastLioFrontendReceivesDerivedConfig) {
    Config config;
    config.frontend_mode = "fast_lio";
    config.lidar_type = "avia";
    config.frontend_lidar_to_body_tx = 0.4;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    const auto* frontend = dynamic_cast<lio::FastLioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);
    EXPECT_EQ(frontend->config().lidar_type, "avia");
    EXPECT_NEAR(frontend->config().T_body_lidar.translation().x(), 0.4, 1e-12);
}

TEST(LioFrontendFactoryTest, FastLioFrontendConsumesRawInputAtAdapterBoundary) {
    Config config;
    config.frontend_mode = "fast_lio";
    config.frontend_blind = 10.0;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    auto* frontend = dynamic_cast<lio::FastLioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);

    frontend->addImu(makeImuSample(1000000000LL));
    frontend->addImu(makeImuSample(1001000000LL));
    const auto output = frontend->addLidar(makeRawLidarFrame());

    EXPECT_FALSE(output.has_value());
    EXPECT_EQ(frontend->imuSamplesSeen(), 2u);
    EXPECT_EQ(frontend->lidarFramesSeen(), 1u);
    EXPECT_EQ(frontend->lastCloudStats().input_points, 1u);
    EXPECT_EQ(frontend->lastCloudStats().output_points, 0u);
    EXPECT_EQ(frontend->lastCloudStats().skipped_blind, 1u);
    EXPECT_EQ(frontend->lastInputImuSamples(), 2u);
    EXPECT_TRUE(frontend->lastInputHadCompleteImuWindow());

    frontend->reset();
    EXPECT_EQ(frontend->imuSamplesSeen(), 0u);
    EXPECT_EQ(frontend->lidarFramesSeen(), 0u);
    EXPECT_EQ(frontend->lastCloudStats().output_points, 0u);
    EXPECT_EQ(frontend->lastInputImuSamples(), 0u);
    EXPECT_FALSE(frontend->lastInputHadCompleteImuWindow());
    EXPECT_FALSE(frontend->predictedState().has_value());
    EXPECT_FALSE(frontend->lastAlignmentStats().valid);
}

TEST(LioFrontendFactoryTest, FastLioFrontendCanReturnPredictionOnlyFrame) {
    Config config;
    config.frontend_mode = "fast_lio";
    config.frontend_prediction_only_output = true;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    auto* frontend = dynamic_cast<lio::FastLioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);

    frontend->addImu(makeImuSample(1000000000LL));
    frontend->addImu(makeImuSample(1001000000LL));
    const auto output = frontend->addLidar(makeRawLidarFrame());

    ASSERT_TRUE(output.has_value());
    EXPECT_TRUE(output->pose_valid);
    ASSERT_TRUE(output->undistorted_cloud);
    EXPECT_EQ(output->undistorted_cloud->size(), 1u);
    ASSERT_TRUE(frontend->predictedState().has_value());
    ASSERT_TRUE(frontend->localMapCloud());
    EXPECT_EQ(frontend->localMapCloud()->size(), 1u);
    EXPECT_FALSE(frontend->lastAlignmentStats().valid);
}

TEST(LioFrontendFactoryTest, FastLioFrontendAppliesLocalMapCorrection) {
    Config config;
    config.frontend_mode = "fast_lio";
    config.frontend_prediction_only_output = true;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    auto* frontend = dynamic_cast<lio::FastLioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);

    frontend->addImu(makeImuSample(1000000000LL));
    frontend->addImu(makeImuSample(1001000000LL));
    ASSERT_TRUE(frontend->addLidar(makeRawLidarFrame()).has_value());

    auto second = makeRawLidarFrame();
    second.points->at(0).x += 0.25f;
    second.stamp_begin.nsec = 1001000000LL;
    second.stamp_end.nsec = 1002000000LL;
    frontend->addImu(makeImuSample(1002000000LL));
    const auto output = frontend->addLidar(second);

    ASSERT_TRUE(output.has_value());
    EXPECT_TRUE(frontend->lastAlignmentStats().valid);
    EXPECT_NEAR(output->T_world_lidar.translation().x(), -0.25, 1e-6);
}

TEST(LioFrontendFactoryTest, FastLioFrontendMatchesCoreBoundaryState) {
    Config config;
    config.frontend_mode = "fast_lio";
    config.frontend_prediction_only_output = true;
    const auto frontend_config = lio::makeLioFrontendConfig(config);
    lio::FastLioFrontend frontend(frontend_config);
    lio::fast_lio::Core core(frontend_config);

    const auto imu0 = makeImuSample(1000000000LL);
    const auto imu1 = makeImuSample(1001000000LL);
    const auto frame = makeRawLidarFrame();
    frontend.addImu(imu0);
    frontend.addImu(imu1);
    core.addImu(imu0);
    core.addImu(imu1);

    const auto frontend_output = frontend.addLidar(frame);
    const auto core_output = core.addLidar(frame);

    ASSERT_EQ(frontend_output.has_value(), core_output.has_value());
    ASSERT_TRUE(frontend_output.has_value());
    EXPECT_EQ(frontend.imuSamplesSeen(), core.imuSamplesSeen());
    EXPECT_EQ(frontend.lidarFramesSeen(), core.lidarFramesSeen());
    EXPECT_EQ(frontend.lastCloudStats().input_points,
              core.lastInputPacket().cloud_stats.input_points);
    EXPECT_EQ(frontend.lastInputImuSamples(),
              core.lastInputPacket().imu_samples.size());
    EXPECT_EQ(frontend.lastInputHadCompleteImuWindow(),
              core.lastInputPacket().has_complete_imu_window);
    EXPECT_NEAR(frontend_output->T_world_lidar.translation().x(),
                core_output->T_world_lidar.translation().x(),
                1e-12);
    ASSERT_TRUE(frontend.localMapCloud());
    ASSERT_TRUE(core.localMapCloud());
    EXPECT_EQ(frontend.localMapCloud()->size(), core.localMapCloud()->size());
    EXPECT_EQ(frontend.lastAlignmentStats().valid,
              core.lastAlignmentStats().valid);
}

TEST(LioFrontendFactoryTest, FastLioDebugCallbacksAreOptIn) {
    Config config;
    config.frontend_mode = "fast_lio";
    config.frontend_prediction_only_output = true;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    auto* frontend = dynamic_cast<lio::FastLioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);

    int odom_count = 0;
    int cloud_count = 0;
    int local_map_count = 0;
    int timing_count = 0;
    lio::LioDebugCallbacks callbacks;
    callbacks.odom = [&](const core::LioFrame&) { ++odom_count; };
    callbacks.deskewed_cloud =
        [&](const core::LioFrame::PointCloud::ConstPtr&) { ++cloud_count; };
    callbacks.local_map =
        [&](const core::LioFrame::PointCloud::ConstPtr&) { ++local_map_count; };
    callbacks.timing = [&](const lio::LioTimingStats&) { ++timing_count; };
    frontend->setDebugCallbacks(callbacks);

    frontend->addImu(makeImuSample(1000000000LL));
    frontend->addImu(makeImuSample(1001000000LL));
    ASSERT_TRUE(frontend->addLidar(makeRawLidarFrame()).has_value());
    EXPECT_EQ(odom_count, 0);
    EXPECT_EQ(cloud_count, 0);
    EXPECT_EQ(local_map_count, 0);
    EXPECT_EQ(timing_count, 0);
}

TEST(LioFrontendFactoryTest, FastLioDebugCallbacksFireWhenEnabled) {
    Config config;
    config.frontend_mode = "fast_lio";
    config.frontend_prediction_only_output = true;
    config.frontend_publish_debug = true;
    config.frontend_debug_publish_odom = true;
    config.frontend_debug_publish_deskewed_cloud = true;
    config.frontend_debug_publish_local_map = true;
    config.frontend_debug_publish_timing = true;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    auto* frontend = dynamic_cast<lio::FastLioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);

    int odom_count = 0;
    int cloud_count = 0;
    int local_map_count = 0;
    int timing_count = 0;
    lio::LioDebugCallbacks callbacks;
    callbacks.odom = [&](const core::LioFrame&) { ++odom_count; };
    callbacks.deskewed_cloud =
        [&](const core::LioFrame::PointCloud::ConstPtr&) { ++cloud_count; };
    callbacks.local_map =
        [&](const core::LioFrame::PointCloud::ConstPtr& cloud) {
            ++local_map_count;
            EXPECT_FALSE(cloud->empty());
        };
    callbacks.timing = [&](const lio::LioTimingStats&) { ++timing_count; };
    frontend->setDebugCallbacks(callbacks);

    frontend->addImu(makeImuSample(1000000000LL));
    frontend->addImu(makeImuSample(1001000000LL));
    ASSERT_TRUE(frontend->addLidar(makeRawLidarFrame()).has_value());
    EXPECT_EQ(odom_count, 1);
    EXPECT_EQ(cloud_count, 1);
    EXPECT_EQ(local_map_count, 1);
    EXPECT_EQ(timing_count, 1);
}
#endif

#ifdef N3MAPPING_BUILD_DLIO_CORE
TEST(LioFrontendFactoryTest, DlioFrontendReceivesDerivedConfig) {
    Config config;
    config.frontend_mode = "dlio";
    config.lidar_type = "ouster";
    config.frontend_imu_to_body_ty = 0.2;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    const auto* frontend = dynamic_cast<lio::DlioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);
    EXPECT_EQ(frontend->config().lidar_type, "ouster");
    EXPECT_NEAR(frontend->config().T_body_imu.translation().y(), 0.2, 1e-12);
}

TEST(LioFrontendFactoryTest, DlioFrontendConsumesRawInputAtAdapterBoundary) {
    Config config;
    config.frontend_mode = "dlio";
    config.dlio_time_encoding = "velodyne";
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    auto* frontend = dynamic_cast<lio::DlioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);

    frontend->addImu(makeImuSample(1000000000LL));
    frontend->addImu(makeImuSample(1001000000LL));
    const auto output = frontend->addLidar(makeRawLidarFrame("livox_custom"));

    EXPECT_FALSE(output.has_value());
    EXPECT_EQ(frontend->imuSamplesSeen(), 2u);
    EXPECT_EQ(frontend->lidarFramesSeen(), 1u);
    EXPECT_EQ(frontend->lastCloudStats().input_points, 1u);
    EXPECT_EQ(frontend->lastCloudStats().output_points, 1u);
    EXPECT_EQ(frontend->lastTimeEncoding(), lio::dlio::TimeEncoding::VelodyneOffsetSeconds);
    EXPECT_EQ(frontend->lastInputImuSamples(), 2u);
    EXPECT_TRUE(frontend->lastInputHadCompleteImuWindow());

    frontend->reset();
    EXPECT_EQ(frontend->imuSamplesSeen(), 0u);
    EXPECT_EQ(frontend->lidarFramesSeen(), 0u);
    EXPECT_EQ(frontend->lastCloudStats().output_points, 0u);
    EXPECT_EQ(frontend->lastInputImuSamples(), 0u);
    EXPECT_FALSE(frontend->lastInputHadCompleteImuWindow());
    EXPECT_FALSE(frontend->predictedState().has_value());
    EXPECT_FALSE(frontend->lastAlignmentStats().valid);
}

TEST(LioFrontendFactoryTest, DlioFrontendCanReturnPredictionOnlyFrame) {
    Config config;
    config.frontend_mode = "dlio";
    config.frontend_prediction_only_output = true;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    auto* frontend = dynamic_cast<lio::DlioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);

    frontend->addImu(makeImuSample(1000000000LL));
    frontend->addImu(makeImuSample(1001000000LL));
    const auto output = frontend->addLidar(makeRawLidarFrame("livox_custom"));

    ASSERT_TRUE(output.has_value());
    EXPECT_TRUE(output->pose_valid);
    ASSERT_TRUE(output->undistorted_cloud);
    EXPECT_EQ(output->undistorted_cloud->size(), 1u);
    ASSERT_TRUE(frontend->predictedState().has_value());
    ASSERT_TRUE(frontend->localMapCloud());
    EXPECT_EQ(frontend->localMapCloud()->size(), 1u);
    ASSERT_TRUE(frontend->denseMapCloud());
    EXPECT_EQ(frontend->denseMapCloud()->size(), 1u);
    EXPECT_TRUE(frontend->lastDenseMapAddResult().accepted);
    EXPECT_FALSE(frontend->lastAlignmentStats().valid);
}

TEST(LioFrontendFactoryTest, DlioFrontendAppliesLocalMapCorrection) {
    Config config;
    config.frontend_mode = "dlio";
    config.frontend_prediction_only_output = true;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    auto* frontend = dynamic_cast<lio::DlioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);

    frontend->addImu(makeImuSample(1000000000LL));
    frontend->addImu(makeImuSample(1001000000LL));
    ASSERT_TRUE(frontend->addLidar(makeRawLidarFrame("livox_custom")).has_value());

    auto second = makeRawLidarFrame("livox_custom");
    second.points->at(0).x += 0.25f;
    second.stamp_begin.nsec = 1001000000LL;
    second.stamp_end.nsec = 1002000000LL;
    frontend->addImu(makeImuSample(1002000000LL));
    const auto output = frontend->addLidar(second);

    ASSERT_TRUE(output.has_value());
    EXPECT_TRUE(frontend->lastAlignmentStats().valid);
    EXPECT_NEAR(output->T_world_lidar.translation().x(), -0.25, 1e-6);
}

TEST(LioFrontendFactoryTest, DlioFrontendMatchesCoreBoundaryState) {
    Config config;
    config.frontend_mode = "dlio";
    config.frontend_prediction_only_output = true;
    config.dlio_time_encoding = "velodyne";
    const auto frontend_config = lio::makeLioFrontendConfig(config);
    lio::DlioFrontend frontend(frontend_config);
    lio::dlio::Core core(frontend_config);

    const auto imu0 = makeImuSample(1000000000LL);
    const auto imu1 = makeImuSample(1001000000LL);
    const auto frame = makeRawLidarFrame("livox_custom");
    frontend.addImu(imu0);
    frontend.addImu(imu1);
    core.addImu(imu0);
    core.addImu(imu1);

    const auto frontend_output = frontend.addLidar(frame);
    const auto core_output = core.addLidar(frame);

    ASSERT_EQ(frontend_output.has_value(), core_output.has_value());
    ASSERT_TRUE(frontend_output.has_value());
    EXPECT_EQ(frontend.imuSamplesSeen(), core.imuSamplesSeen());
    EXPECT_EQ(frontend.lidarFramesSeen(), core.lidarFramesSeen());
    EXPECT_EQ(frontend.lastCloudStats().input_points,
              core.lastInputPacket().cloud_stats.input_points);
    EXPECT_EQ(frontend.lastTimeEncoding(), core.lastInputPacket().time_encoding);
    EXPECT_EQ(frontend.lastInputImuSamples(),
              core.lastInputPacket().imu_samples.size());
    EXPECT_EQ(frontend.lastInputHadCompleteImuWindow(),
              core.lastInputPacket().has_complete_imu_window);
    EXPECT_NEAR(frontend_output->T_world_lidar.translation().x(),
                core_output->T_world_lidar.translation().x(),
                1e-12);
    ASSERT_TRUE(frontend.localMapCloud());
    ASSERT_TRUE(core.localMapCloud());
    EXPECT_EQ(frontend.localMapCloud()->size(), core.localMapCloud()->size());
    ASSERT_TRUE(frontend.denseMapCloud());
    ASSERT_TRUE(core.denseMapCloud());
    EXPECT_EQ(frontend.denseMapCloud()->size(), core.denseMapCloud()->size());
    EXPECT_EQ(frontend.lastDenseMapAddResult().accepted,
              core.lastDenseMapAddResult().accepted);
    EXPECT_EQ(frontend.lastAlignmentStats().valid,
              core.lastAlignmentStats().valid);
}

TEST(LioFrontendFactoryTest, DlioDebugCallbacksAreOptIn) {
    Config config;
    config.frontend_mode = "dlio";
    config.frontend_prediction_only_output = true;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    auto* frontend = dynamic_cast<lio::DlioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);

    int odom_count = 0;
    int cloud_count = 0;
    int local_map_count = 0;
    int timing_count = 0;
    lio::LioDebugCallbacks callbacks;
    callbacks.odom = [&](const core::LioFrame&) { ++odom_count; };
    callbacks.deskewed_cloud =
        [&](const core::LioFrame::PointCloud::ConstPtr&) { ++cloud_count; };
    callbacks.local_map =
        [&](const core::LioFrame::PointCloud::ConstPtr&) { ++local_map_count; };
    callbacks.timing = [&](const lio::LioTimingStats&) { ++timing_count; };
    frontend->setDebugCallbacks(callbacks);

    frontend->addImu(makeImuSample(1000000000LL));
    frontend->addImu(makeImuSample(1001000000LL));
    ASSERT_TRUE(frontend->addLidar(makeRawLidarFrame("livox_custom")).has_value());
    EXPECT_EQ(odom_count, 0);
    EXPECT_EQ(cloud_count, 0);
    EXPECT_EQ(local_map_count, 0);
    EXPECT_EQ(timing_count, 0);
}

TEST(LioFrontendFactoryTest, DlioDebugCallbacksFireWhenEnabled) {
    Config config;
    config.frontend_mode = "dlio";
    config.frontend_prediction_only_output = true;
    config.frontend_publish_debug = true;
    config.frontend_debug_publish_odom = true;
    config.frontend_debug_publish_deskewed_cloud = true;
    config.frontend_debug_publish_local_map = true;
    config.frontend_debug_publish_timing = true;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    auto* frontend = dynamic_cast<lio::DlioFrontend*>(result.frontend.get());
    ASSERT_NE(frontend, nullptr);

    int odom_count = 0;
    int cloud_count = 0;
    int local_map_count = 0;
    int timing_count = 0;
    lio::LioDebugCallbacks callbacks;
    callbacks.odom = [&](const core::LioFrame&) { ++odom_count; };
    callbacks.deskewed_cloud =
        [&](const core::LioFrame::PointCloud::ConstPtr&) { ++cloud_count; };
    callbacks.local_map =
        [&](const core::LioFrame::PointCloud::ConstPtr& cloud) {
            ++local_map_count;
            EXPECT_FALSE(cloud->empty());
        };
    callbacks.timing = [&](const lio::LioTimingStats&) { ++timing_count; };
    frontend->setDebugCallbacks(callbacks);

    frontend->addImu(makeImuSample(1000000000LL));
    frontend->addImu(makeImuSample(1001000000LL));
    ASSERT_TRUE(frontend->addLidar(makeRawLidarFrame("livox_custom")).has_value());
    EXPECT_EQ(odom_count, 1);
    EXPECT_EQ(cloud_count, 1);
    EXPECT_EQ(local_map_count, 1);
    EXPECT_EQ(timing_count, 1);
}
#endif

}  // namespace test
}  // namespace n3mapping
