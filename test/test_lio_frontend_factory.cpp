#include <cmath>

#include <gtest/gtest.h>

#ifdef N3MAPPING_BUILD_DLIO_CORE
#include "n3mapping/lio/dlio_frontend.h"
#endif
#include "n3mapping/lio/external_frontend.h"
#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
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
#endif

}  // namespace

TEST(LioFrontendFactoryTest, CreatesExternalFrontendByDefault) {
    Config config;
    auto result = lio::createLioFrontend(config);
    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_EQ(result.mode, lio::FrontendMode::External);
    EXPECT_NE(dynamic_cast<lio::ExternalLioFrontend*>(result.frontend.get()), nullptr);
}

TEST(LioFrontendFactoryTest, ParsesAliases) {
    EXPECT_EQ(lio::parseFrontendMode("external"), lio::FrontendMode::External);
    EXPECT_EQ(lio::parseFrontendMode("FAST_LIO2"), lio::FrontendMode::FastLio);
    EXPECT_EQ(lio::parseFrontendMode("fastlio"), lio::FrontendMode::FastLio);
    EXPECT_EQ(lio::parseFrontendMode("DLIO"), lio::FrontendMode::Dlio);
    EXPECT_EQ(lio::parseFrontendMode("bad"), lio::FrontendMode::Unknown);
}

TEST(LioFrontendFactoryTest, BuiltinFrontendsAreExplicitlyUnsupportedForNow) {
    Config config;
    config.frontend_mode = "fast_lio";
    auto fast_lio = lio::createLioFrontend(config);
#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
    EXPECT_TRUE(fast_lio.ok()) << fast_lio.error;
#else
    EXPECT_FALSE(fast_lio.ok());
    EXPECT_NE(fast_lio.error.find("fast_lio_core"), std::string::npos);
#endif
    EXPECT_EQ(fast_lio.mode, lio::FrontendMode::FastLio);

    config.frontend_mode = "dlio";
    auto dlio = lio::createLioFrontend(config);
#ifdef N3MAPPING_BUILD_DLIO_CORE
    EXPECT_TRUE(dlio.ok()) << dlio.error;
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
    config.dlio_time_encoding = "velodyne";
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
    EXPECT_EQ(frontend_config.dlio_time_encoding, "velodyne");
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

    frontend->addImu(core::ImuSample{});
    const auto output = frontend->addLidar(makeRawLidarFrame());

    EXPECT_FALSE(output.has_value());
    EXPECT_EQ(frontend->imuSamplesSeen(), 1u);
    EXPECT_EQ(frontend->lidarFramesSeen(), 1u);
    EXPECT_EQ(frontend->lastCloudStats().input_points, 1u);
    EXPECT_EQ(frontend->lastCloudStats().output_points, 0u);
    EXPECT_EQ(frontend->lastCloudStats().skipped_blind, 1u);

    frontend->reset();
    EXPECT_EQ(frontend->imuSamplesSeen(), 0u);
    EXPECT_EQ(frontend->lidarFramesSeen(), 0u);
    EXPECT_EQ(frontend->lastCloudStats().output_points, 0u);
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

    frontend->addImu(core::ImuSample{});
    const auto output = frontend->addLidar(makeRawLidarFrame("livox_custom"));

    EXPECT_FALSE(output.has_value());
    EXPECT_EQ(frontend->imuSamplesSeen(), 1u);
    EXPECT_EQ(frontend->lidarFramesSeen(), 1u);
    EXPECT_EQ(frontend->lastCloudStats().input_points, 1u);
    EXPECT_EQ(frontend->lastCloudStats().output_points, 1u);
    EXPECT_EQ(frontend->lastTimeEncoding(), lio::dlio::TimeEncoding::VelodyneOffsetSeconds);

    frontend->reset();
    EXPECT_EQ(frontend->imuSamplesSeen(), 0u);
    EXPECT_EQ(frontend->lidarFramesSeen(), 0u);
    EXPECT_EQ(frontend->lastCloudStats().output_points, 0u);
}
#endif

}  // namespace test
}  // namespace n3mapping
