#include <gtest/gtest.h>

#include "n3mapping/lio/fast_lio_settings.h"

namespace n3mapping {
namespace test {

TEST(FastLioSettingsTest, ParsesLidarAliases) {
    EXPECT_EQ(lio::fast_lio::parseLidarType("avia"),
              lio::fast_lio::LidarType::Avia);
    EXPECT_EQ(lio::fast_lio::parseLidarType("MID360"),
              lio::fast_lio::LidarType::Avia);
    EXPECT_EQ(lio::fast_lio::parseLidarType("velo16"),
              lio::fast_lio::LidarType::Velodyne);
    EXPECT_EQ(lio::fast_lio::parseLidarType("ouster"),
              lio::fast_lio::LidarType::Ouster);
    EXPECT_EQ(lio::fast_lio::parseLidarType("marsim"),
              lio::fast_lio::LidarType::Marsim);
    EXPECT_EQ(lio::fast_lio::parseLidarType("generic"),
              lio::fast_lio::LidarType::Generic);
}

TEST(FastLioSettingsTest, BuildsCloudAdapterOptionsFromFrontendConfig) {
    lio::LioFrontendConfig config;
    config.lidar_type = "mid360";
    config.point_filter_num = 3;
    config.scan_lines = 32;
    config.blind = 0.2;
    config.max_abs_coordinate = 50.0;
    config.alignment_max_correspondence_distance = 0.8;

    const auto settings = lio::fast_lio::makeSettings(config);
    EXPECT_EQ(settings.lidar_type, lio::fast_lio::LidarType::Avia);
    EXPECT_EQ(settings.point_filter_num, 3u);
    EXPECT_EQ(settings.scan_lines, 32u);
    EXPECT_DOUBLE_EQ(settings.blind, 0.2);
    EXPECT_DOUBLE_EQ(settings.max_abs_coordinate, 50.0);
    EXPECT_DOUBLE_EQ(settings.alignment_max_correspondence_distance, 0.8);

    const auto options = lio::fast_lio::makeCloudAdapterOptions(settings);
    EXPECT_EQ(options.point_filter_num, 3u);
    EXPECT_EQ(options.scan_lines, 32u);
    EXPECT_DOUBLE_EQ(options.blind, 0.2);
    EXPECT_DOUBLE_EQ(options.max_abs_coordinate, 50.0);
}

}  // namespace test
}  // namespace n3mapping
