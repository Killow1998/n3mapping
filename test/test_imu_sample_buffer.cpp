#include <gtest/gtest.h>

#include "n3mapping/lio/imu_sample_buffer.h"

namespace n3mapping {
namespace test {
namespace {

core::ImuSample makeSample(int64_t stamp_nsec, double gyro_z = 0.0) {
    core::ImuSample sample;
    sample.stamp.nsec = stamp_nsec;
    sample.angular_velocity.z() = gyro_z;
    return sample;
}

}  // namespace

TEST(ImuSampleBufferTest, StoresLatestSample) {
    lio::ImuSampleBuffer buffer;
    EXPECT_TRUE(buffer.empty());
    EXPECT_FALSE(buffer.latest().has_value());

    buffer.add(makeSample(10, 1.0));
    buffer.add(makeSample(20, 2.0));

    ASSERT_TRUE(buffer.latest().has_value());
    EXPECT_EQ(buffer.latest()->stamp.nsec, 20);
    EXPECT_NEAR(buffer.latest()->angular_velocity.z(), 2.0, 1e-12);
}

TEST(ImuSampleBufferTest, DropsOldSamplesWhenCapacityIsExceeded) {
    lio::ImuSampleBuffer buffer(2);
    buffer.add(makeSample(10));
    buffer.add(makeSample(20));
    buffer.add(makeSample(30));

    EXPECT_EQ(buffer.size(), 2u);
    const auto samples = buffer.samplesInRange(0, 100);
    ASSERT_EQ(samples.size(), 2u);
    EXPECT_EQ(samples[0].stamp.nsec, 20);
    EXPECT_EQ(samples[1].stamp.nsec, 30);
}

TEST(ImuSampleBufferTest, ReturnsClosedTimeRange) {
    lio::ImuSampleBuffer buffer;
    buffer.add(makeSample(10));
    buffer.add(makeSample(20));
    buffer.add(makeSample(30));

    const auto samples = buffer.samplesInRange(10, 20);
    ASSERT_EQ(samples.size(), 2u);
    EXPECT_EQ(samples[0].stamp.nsec, 10);
    EXPECT_EQ(samples[1].stamp.nsec, 20);
    EXPECT_TRUE(buffer.samplesInRange(30, 10).empty());
}

TEST(ImuSampleBufferTest, ClearResetsState) {
    lio::ImuSampleBuffer buffer;
    buffer.add(makeSample(10));
    buffer.clear();

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_FALSE(buffer.latest().has_value());
}

}  // namespace test
}  // namespace n3mapping
