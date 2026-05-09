#include <gtest/gtest.h>

#include "n3mapping/lio/dlio_scan_timing.h"

namespace n3mapping {
namespace test {
namespace {

core::RawLidarFrame makeFrame() {
    core::RawLidarFrame frame;
    frame.stamp_begin.nsec = 1000000000LL;
    frame.stamp_end.nsec = 1010000000LL;
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();
    for (int i = 0; i < 4; ++i) {
        pcl::PointXYZI point;
        point.x = static_cast<float>(i);
        frame.points->push_back(point);
    }
    frame.points->width = static_cast<uint32_t>(frame.points->size());
    frame.points->height = 1;
    frame.points->is_dense = true;
    return frame;
}

}  // namespace

TEST(DlioScanTimingTest, ComputesSortedUniquePointTimestamps) {
    auto frame = makeFrame();
    frame.point_time_offsets_ns = {3000000u, 1000000u, 3000000u, 9000000u};

    const auto timing = lio::dlio::computeScanTiming(frame);

    EXPECT_TRUE(timing.valid);
    EXPECT_TRUE(timing.has_point_timing);
    ASSERT_EQ(timing.unique_point_timestamps.size(), 3u);
    EXPECT_NEAR(timing.unique_point_timestamps[0], 1.001, 1e-12);
    EXPECT_NEAR(timing.unique_point_timestamps[1], 1.003, 1e-12);
    EXPECT_NEAR(timing.unique_point_timestamps[2], 1.009, 1e-12);
    EXPECT_NEAR(timing.stamp_median, 1.003, 1e-12);
}

TEST(DlioScanTimingTest, FallsBackToSweepMidpointWithoutPointOffsets) {
    const auto timing = lio::dlio::computeScanTiming(makeFrame());

    EXPECT_TRUE(timing.valid);
    EXPECT_FALSE(timing.has_point_timing);
    ASSERT_EQ(timing.unique_point_timestamps.size(), 1u);
    EXPECT_NEAR(timing.stamp_begin, 1.0, 1e-12);
    EXPECT_NEAR(timing.stamp_end, 1.01, 1e-12);
    EXPECT_NEAR(timing.stamp_median, 1.005, 1e-12);
}

TEST(DlioScanTimingTest, KeepsReversedFrameRangeUsable) {
    auto frame = makeFrame();
    frame.stamp_begin.nsec = 1010000000LL;
    frame.stamp_end.nsec = 1000000000LL;

    const auto timing = lio::dlio::computeScanTiming(frame);

    EXPECT_TRUE(timing.valid);
    EXPECT_NEAR(timing.stamp_begin, 1.0, 1e-12);
    EXPECT_NEAR(timing.stamp_end, 1.01, 1e-12);
    EXPECT_NEAR(timing.stamp_median, 1.005, 1e-12);
}

TEST(DlioScanTimingTest, EmptyCloudIsInvalidButHasSweepMidpoint) {
    core::RawLidarFrame frame;
    frame.stamp_begin.nsec = 2000000000LL;
    frame.stamp_end.nsec = 2010000000LL;
    frame.points = pcl::make_shared<core::RawLidarFrame::PointCloud>();

    const auto timing = lio::dlio::computeScanTiming(frame);

    EXPECT_FALSE(timing.valid);
    EXPECT_FALSE(timing.has_point_timing);
    EXPECT_TRUE(timing.unique_point_timestamps.empty());
    EXPECT_NEAR(timing.stamp_median, 2.005, 1e-12);
}

}  // namespace test
}  // namespace n3mapping
