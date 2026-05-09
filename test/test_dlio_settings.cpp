#include <gtest/gtest.h>

#include "n3mapping/lio/dlio_settings.h"

namespace n3mapping {
namespace test {

TEST(DlioSettingsTest, ParsesSensorAliases) {
    EXPECT_EQ(lio::dlio::parseSensorType("ouster"), lio::dlio::SensorType::Ouster);
    EXPECT_EQ(lio::dlio::parseSensorType("VELODYNE"), lio::dlio::SensorType::Velodyne);
    EXPECT_EQ(lio::dlio::parseSensorType("hesai"), lio::dlio::SensorType::Hesai);
    EXPECT_EQ(lio::dlio::parseSensorType("livox"), lio::dlio::SensorType::Livox);
    EXPECT_EQ(lio::dlio::parseSensorType("mid360"), lio::dlio::SensorType::Livox);
    EXPECT_EQ(lio::dlio::parseSensorType("generic"), lio::dlio::SensorType::Unknown);
}

TEST(DlioSettingsTest, ParsesTimeEncodingAliases) {
    EXPECT_EQ(lio::dlio::parseTimeEncoding("auto"), lio::dlio::TimeEncoding::Auto);
    EXPECT_EQ(lio::dlio::parseTimeEncoding("ouster_ns"),
              lio::dlio::TimeEncoding::OusterOffsetNs);
    EXPECT_EQ(lio::dlio::parseTimeEncoding("velodyne_seconds"),
              lio::dlio::TimeEncoding::VelodyneOffsetSeconds);
    EXPECT_EQ(lio::dlio::parseTimeEncoding("livox_ns"),
              lio::dlio::TimeEncoding::LivoxOffsetNs);
    EXPECT_EQ(lio::dlio::parseTimeEncoding("unknown"),
              lio::dlio::TimeEncoding::Auto);
}

TEST(DlioSettingsTest, BuildsCloudAdapterOptionsFromFrontendConfig) {
    lio::LioFrontendConfig config;
    config.lidar_type = "mid360";
    config.dlio_time_encoding = "livox";
    config.blind = 0.3;
    config.max_abs_coordinate = 42.0;
    config.alignment_max_correspondence_distance = 0.7;

    const auto settings = lio::dlio::makeSettings(config);
    EXPECT_EQ(settings.sensor, lio::dlio::SensorType::Livox);
    EXPECT_EQ(settings.time_encoding, lio::dlio::TimeEncoding::LivoxOffsetNs);
    EXPECT_DOUBLE_EQ(settings.blind, 0.3);
    EXPECT_DOUBLE_EQ(settings.max_abs_coordinate, 42.0);
    EXPECT_DOUBLE_EQ(settings.alignment_max_correspondence_distance, 0.7);

    const auto options = lio::dlio::makeCloudAdapterOptions(settings);
    EXPECT_EQ(options.time_encoding, lio::dlio::TimeEncoding::LivoxOffsetNs);
    EXPECT_DOUBLE_EQ(options.blind, 0.3);
    EXPECT_DOUBLE_EQ(options.max_abs_coordinate, 42.0);
}

}  // namespace test
}  // namespace n3mapping
