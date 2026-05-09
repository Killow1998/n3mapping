#include <limits>

#include <gtest/gtest.h>

#include "n3mapping/lio/fast_lio_cloud_adapter.h"

namespace n3mapping {
namespace test {
namespace {

core::RawLidarFrame makeRawFrame() {
    core::RawLidarFrame frame;
    frame.source_format = "pointcloud2";
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();

    pcl::PointXYZI point;
    point.x = 1.0f;
    point.y = 2.0f;
    point.z = 3.0f;
    point.intensity = 10.0f;
    frame.points->push_back(point);
    frame.point_time_offsets_ns.push_back(1500000u);
    frame.point_lines.push_back(3u);

    point.x = 2.0f;
    point.y = 0.0f;
    point.z = 0.0f;
    point.intensity = 20.0f;
    frame.points->push_back(point);
    frame.point_time_offsets_ns.push_back(2500000u);
    frame.point_lines.push_back(4u);

    frame.points->width = static_cast<uint32_t>(frame.points->size());
    frame.points->height = 1;
    frame.points->is_dense = true;
    return frame;
}

}  // namespace

TEST(FastLioCloudAdapterTest, ConvertsTimingToCurvatureMilliseconds) {
    const auto frame = makeRawFrame();
    const auto result = lio::fast_lio::cloudFromRawLidar(frame);

    ASSERT_TRUE(result.cloud);
    ASSERT_EQ(result.cloud->size(), 2u);
    EXPECT_EQ(result.stats.input_points, 2u);
    EXPECT_EQ(result.stats.output_points, 2u);
    EXPECT_NEAR(result.cloud->at(0).x, 1.0f, 1e-6f);
    EXPECT_NEAR(result.cloud->at(0).intensity, 10.0f, 1e-6f);
    EXPECT_NEAR(result.cloud->at(0).curvature, 1.5f, 1e-6f);
    EXPECT_NEAR(result.cloud->at(1).curvature, 2.5f, 1e-6f);
    EXPECT_EQ(result.cloud->width, 2u);
    EXPECT_EQ(result.cloud->height, 1u);
}

TEST(FastLioCloudAdapterTest, FiltersInvalidPointsWithoutRosTypes) {
    auto frame = makeRawFrame();
    frame.points->at(0).x = std::numeric_limits<float>::quiet_NaN();
    frame.point_lines[1] = 200u;

    lio::fast_lio::CloudAdapterOptions options;
    options.scan_lines = 128;
    const auto result = lio::fast_lio::cloudFromRawLidar(frame, options);

    ASSERT_TRUE(result.cloud);
    EXPECT_TRUE(result.cloud->empty());
    EXPECT_EQ(result.stats.skipped_non_finite, 1u);
    EXPECT_EQ(result.stats.skipped_invalid_line, 1u);
}

TEST(FastLioCloudAdapterTest, AppliesPointFilterAndBlindRange) {
    auto frame = makeRawFrame();
    frame.points->at(0).x = 0.05f;
    frame.points->at(0).y = 0.0f;
    frame.points->at(0).z = 0.0f;

    lio::fast_lio::CloudAdapterOptions options;
    options.point_filter_num = 1;
    options.blind = 0.1;
    const auto blind_filtered = lio::fast_lio::cloudFromRawLidar(frame, options);
    ASSERT_TRUE(blind_filtered.cloud);
    ASSERT_EQ(blind_filtered.cloud->size(), 1u);
    EXPECT_EQ(blind_filtered.stats.skipped_blind, 1u);
    EXPECT_NEAR(blind_filtered.cloud->at(0).intensity, 20.0f, 1e-6f);

    options.blind = 0.0;
    options.point_filter_num = 2;
    const auto downsampled = lio::fast_lio::cloudFromRawLidar(frame, options);
    ASSERT_TRUE(downsampled.cloud);
    ASSERT_EQ(downsampled.cloud->size(), 1u);
    EXPECT_EQ(downsampled.stats.skipped_by_filter, 1u);
    EXPECT_NEAR(downsampled.cloud->at(0).intensity, 10.0f, 1e-6f);
}

}  // namespace test
}  // namespace n3mapping
