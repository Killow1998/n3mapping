#include <limits>

#include <gtest/gtest.h>

#include "n3mapping/lio/dlio_cloud_adapter.h"

namespace n3mapping {
namespace test {
namespace {

core::RawLidarFrame makeRawFrame(const std::string& source_format) {
    core::RawLidarFrame frame;
    frame.source_format = source_format;
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();

    pcl::PointXYZI point;
    point.x = 1.0f;
    point.y = 2.0f;
    point.z = 3.0f;
    point.intensity = 11.0f;
    frame.points->push_back(point);
    frame.point_time_offsets_ns.push_back(1000000u);

    point.x = 2.0f;
    point.y = 3.0f;
    point.z = 4.0f;
    point.intensity = 22.0f;
    frame.points->push_back(point);
    frame.point_time_offsets_ns.push_back(2500000u);

    frame.points->width = static_cast<uint32_t>(frame.points->size());
    frame.points->height = 1;
    frame.points->is_dense = true;
    return frame;
}

}  // namespace

TEST(DlioCloudAdapterTest, UsesOusterNanosecondOffsetsByDefault) {
    const auto result = lio::dlio::cloudFromRawLidar(makeRawFrame("pointcloud2"));

    ASSERT_TRUE(result.cloud);
    ASSERT_EQ(result.cloud->size(), 2u);
    EXPECT_EQ(result.resolved_time_encoding, lio::dlio::TimeEncoding::OusterOffsetNs);
    EXPECT_EQ(result.cloud->at(0).t, 1000000u);
    EXPECT_EQ(result.cloud->at(1).t, 2500000u);
    EXPECT_NEAR(result.cloud->at(1).intensity, 22.0f, 1e-6f);
}

TEST(DlioCloudAdapterTest, ConvertsVelodyneOffsetsToSeconds) {
    lio::dlio::CloudAdapterOptions options;
    options.time_encoding = lio::dlio::TimeEncoding::VelodyneOffsetSeconds;
    const auto result = lio::dlio::cloudFromRawLidar(makeRawFrame("pointcloud2"),
                                                     options);

    ASSERT_TRUE(result.cloud);
    ASSERT_EQ(result.cloud->size(), 2u);
    EXPECT_EQ(result.resolved_time_encoding,
              lio::dlio::TimeEncoding::VelodyneOffsetSeconds);
    EXPECT_NEAR(result.cloud->at(0).time, 0.001f, 1e-9f);
    EXPECT_NEAR(result.cloud->at(1).time, 0.0025f, 1e-9f);
}

TEST(DlioCloudAdapterTest, AutoSelectsLivoxNanosecondTimestamp) {
    const auto result = lio::dlio::cloudFromRawLidar(makeRawFrame("livox_custom"));

    ASSERT_TRUE(result.cloud);
    ASSERT_EQ(result.cloud->size(), 2u);
    EXPECT_EQ(result.resolved_time_encoding, lio::dlio::TimeEncoding::LivoxOffsetNs);
    EXPECT_NEAR(result.cloud->at(0).timestamp, 1000000.0, 1e-12);
    EXPECT_NEAR(result.cloud->at(1).timestamp, 2500000.0, 1e-12);
}

TEST(DlioCloudAdapterTest, FiltersInvalidAndBlindPoints) {
    auto frame = makeRawFrame("pointcloud2");
    frame.points->at(0).x = std::numeric_limits<float>::quiet_NaN();
    frame.points->at(1).x = 0.01f;
    frame.points->at(1).y = 0.0f;
    frame.points->at(1).z = 0.0f;

    lio::dlio::CloudAdapterOptions options;
    options.blind = 0.1;
    const auto result = lio::dlio::cloudFromRawLidar(frame, options);

    ASSERT_TRUE(result.cloud);
    EXPECT_TRUE(result.cloud->empty());
    EXPECT_EQ(result.stats.skipped_non_finite, 1u);
    EXPECT_EQ(result.stats.skipped_blind, 1u);
}

}  // namespace test
}  // namespace n3mapping
