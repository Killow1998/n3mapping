#include <gtest/gtest.h>

#include "n3mapping/lio/fast_lio_input_adapter.h"

namespace n3mapping {
namespace test {
namespace {

core::RawLidarFrame makeFrame() {
    core::RawLidarFrame frame;
    frame.stamp_begin.nsec = 2000000000LL;
    frame.stamp_end.nsec = 2010000000LL;
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();

    pcl::PointXYZI point;
    point.x = 2.0f;
    point.y = 0.0f;
    point.z = 0.0f;
    point.intensity = 7.0f;
    frame.points->push_back(point);
    frame.point_time_offsets_ns.push_back(3000000u);
    frame.point_lines.push_back(4u);
    frame.points->width = 1;
    frame.points->height = 1;
    frame.points->is_dense = true;
    return frame;
}

core::ImuSample makeImu(int64_t stamp_nsec) {
    core::ImuSample sample;
    sample.stamp.nsec = stamp_nsec;
    return sample;
}

}  // namespace

TEST(FastLioInputAdapterTest, BuildsCloudAndImuWindow) {
    lio::ImuSampleBuffer imu_buffer;
    imu_buffer.add(makeImu(2000000000LL));
    imu_buffer.add(makeImu(2005000000LL));
    imu_buffer.add(makeImu(2010000000LL));

    const auto packet = lio::fast_lio::buildInputPacket(makeFrame(), imu_buffer);

    ASSERT_TRUE(packet.cloud);
    ASSERT_EQ(packet.cloud->size(), 1u);
    EXPECT_NEAR(packet.cloud->at(0).curvature, 3.0f, 1e-6f);
    EXPECT_EQ(packet.cloud_stats.input_points, 1u);
    EXPECT_EQ(packet.cloud_stats.output_points, 1u);
    ASSERT_EQ(packet.imu_samples.size(), 3u);
    EXPECT_TRUE(packet.has_complete_imu_window);
}

TEST(FastLioInputAdapterTest, MarksIncompleteImuWindow) {
    lio::ImuSampleBuffer imu_buffer;
    imu_buffer.add(makeImu(2005000000LL));

    const auto packet = lio::fast_lio::buildInputPacket(makeFrame(), imu_buffer);

    ASSERT_EQ(packet.imu_samples.size(), 1u);
    EXPECT_FALSE(packet.has_complete_imu_window);
}

TEST(FastLioInputAdapterTest, AppliesCloudOptions) {
    lio::ImuSampleBuffer imu_buffer;
    lio::fast_lio::CloudAdapterOptions options;
    options.scan_lines = 2;

    const auto packet = lio::fast_lio::buildInputPacket(makeFrame(), imu_buffer, options);

    ASSERT_TRUE(packet.cloud);
    EXPECT_TRUE(packet.cloud->empty());
    EXPECT_EQ(packet.cloud_stats.skipped_invalid_line, 1u);
}

}  // namespace test
}  // namespace n3mapping
