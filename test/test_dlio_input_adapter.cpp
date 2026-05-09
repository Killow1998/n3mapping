#include <gtest/gtest.h>

#include "n3mapping/lio/dlio_input_adapter.h"

namespace n3mapping {
namespace test {
namespace {

core::RawLidarFrame makeFrame() {
    core::RawLidarFrame frame;
    frame.source_format = "livox_custom";
    frame.stamp_begin.nsec = 1000000000LL;
    frame.stamp_end.nsec = 1010000000LL;
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();

    pcl::PointXYZI point;
    point.x = 1.0f;
    point.y = 0.0f;
    point.z = 0.0f;
    point.intensity = 9.0f;
    frame.points->push_back(point);
    frame.point_time_offsets_ns.push_back(5000000u);
    frame.points->width = 1;
    frame.points->height = 1;
    frame.points->is_dense = true;
    return frame;
}

core::ImuSample makeImu(int64_t stamp_nsec) {
    core::ImuSample sample;
    sample.stamp.nsec = stamp_nsec;
    sample.angular_velocity.x() = static_cast<double>(stamp_nsec);
    return sample;
}

}  // namespace

TEST(DlioInputAdapterTest, BuildsCloudAndImuWindow) {
    lio::ImuSampleBuffer imu_buffer;
    imu_buffer.add(makeImu(999000000LL));
    imu_buffer.add(makeImu(1000000000LL));
    imu_buffer.add(makeImu(1005000000LL));
    imu_buffer.add(makeImu(1010000000LL));
    imu_buffer.add(makeImu(1011000000LL));

    const auto packet = lio::dlio::buildInputPacket(makeFrame(), imu_buffer);

    EXPECT_EQ(packet.stamp_begin.nsec, 1000000000LL);
    EXPECT_EQ(packet.stamp_end.nsec, 1010000000LL);
    ASSERT_TRUE(packet.cloud);
    ASSERT_EQ(packet.cloud->size(), 1u);
    EXPECT_EQ(packet.cloud_stats.input_points, 1u);
    EXPECT_EQ(packet.cloud_stats.output_points, 1u);
    EXPECT_EQ(packet.time_encoding, lio::dlio::TimeEncoding::LivoxOffsetNs);
    ASSERT_EQ(packet.imu_samples.size(), 3u);
    EXPECT_EQ(packet.imu_samples.front().stamp.nsec, 1000000000LL);
    EXPECT_EQ(packet.imu_samples.back().stamp.nsec, 1010000000LL);
    EXPECT_TRUE(packet.has_complete_imu_window);
}

TEST(DlioInputAdapterTest, MarksIncompleteImuWindow) {
    lio::ImuSampleBuffer imu_buffer;
    imu_buffer.add(makeImu(1005000000LL));

    const auto packet = lio::dlio::buildInputPacket(makeFrame(), imu_buffer);

    ASSERT_EQ(packet.imu_samples.size(), 1u);
    EXPECT_FALSE(packet.has_complete_imu_window);
}

TEST(DlioInputAdapterTest, AppliesCloudOptions) {
    lio::ImuSampleBuffer imu_buffer;
    lio::dlio::CloudAdapterOptions options;
    options.blind = 2.0;
    options.time_encoding = lio::dlio::TimeEncoding::VelodyneOffsetSeconds;

    const auto packet = lio::dlio::buildInputPacket(makeFrame(), imu_buffer, options);

    ASSERT_TRUE(packet.cloud);
    EXPECT_TRUE(packet.cloud->empty());
    EXPECT_EQ(packet.cloud_stats.skipped_blind, 1u);
    EXPECT_EQ(packet.time_encoding, lio::dlio::TimeEncoding::VelodyneOffsetSeconds);
}

}  // namespace test
}  // namespace n3mapping
