#include <cmath>

#include <gtest/gtest.h>

#include "n3mapping/lio/imu_propagator.h"

namespace n3mapping {
namespace test {
namespace {

core::ImuSample makeSample(int64_t stamp_nsec,
                           const Eigen::Vector3d& accel,
                           const Eigen::Vector3d& gyro) {
    core::ImuSample sample;
    sample.stamp.nsec = stamp_nsec;
    sample.linear_accel = accel;
    sample.angular_velocity = gyro;
    return sample;
}

}  // namespace

TEST(ImuPropagatorTest, ReturnsInvalidForEmptySamples) {
    const auto propagated = lio::propagateImu({}, lio::ImuPropagationState{});
    EXPECT_FALSE(propagated.valid);
    EXPECT_EQ(propagated.used_samples, 0u);
}

TEST(ImuPropagatorTest, IntegratesConstantAcceleration) {
    std::vector<core::ImuSample> samples;
    samples.push_back(makeSample(0, Eigen::Vector3d(1.0, 0.0, 0.0),
                                 Eigen::Vector3d::Zero()));
    samples.push_back(makeSample(1000000000LL, Eigen::Vector3d(1.0, 0.0, 0.0),
                                 Eigen::Vector3d::Zero()));

    const auto propagated = lio::propagateImu(samples, lio::ImuPropagationState{});

    ASSERT_TRUE(propagated.valid);
    EXPECT_EQ(propagated.used_samples, 2u);
    EXPECT_NEAR(propagated.position.x(), 0.5, 1e-12);
    EXPECT_NEAR(propagated.velocity.x(), 1.0, 1e-12);
    EXPECT_NEAR(propagated.position.y(), 0.0, 1e-12);
}

TEST(ImuPropagatorTest, IntegratesConstantYawRate) {
    std::vector<core::ImuSample> samples;
    samples.push_back(makeSample(0, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d(0.0, 0.0, M_PI_2)));
    samples.push_back(makeSample(1000000000LL, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d(0.0, 0.0, M_PI_2)));

    const auto propagated = lio::propagateImu(samples, lio::ImuPropagationState{});
    const Eigen::Vector3d x_axis =
        propagated.orientation * Eigen::Vector3d::UnitX();

    ASSERT_TRUE(propagated.valid);
    EXPECT_NEAR(x_axis.x(), 0.0, 1e-9);
    EXPECT_NEAR(x_axis.y(), 1.0, 1e-9);
    EXPECT_NEAR(x_axis.z(), 0.0, 1e-9);
}

TEST(ImuPropagatorTest, AppliesWorldGravityTerm) {
    std::vector<core::ImuSample> samples;
    samples.push_back(makeSample(0, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero()));
    samples.push_back(makeSample(1000000000LL, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero()));

    lio::ImuPropagationOptions options;
    options.gravity_world = Eigen::Vector3d(0.0, 0.0, -9.8);
    const auto propagated = lio::propagateImu(samples, lio::ImuPropagationState{},
                                              options);

    ASSERT_TRUE(propagated.valid);
    EXPECT_NEAR(propagated.velocity.z(), -9.8, 1e-12);
    EXPECT_NEAR(propagated.position.z(), -4.9, 1e-12);
}

TEST(ImuPropagatorTest, SkipsNonIncreasingSamples) {
    std::vector<core::ImuSample> samples;
    samples.push_back(makeSample(0, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero()));
    samples.push_back(makeSample(0, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero()));
    samples.push_back(makeSample(1000000000LL, Eigen::Vector3d(1.0, 0.0, 0.0),
                                 Eigen::Vector3d::Zero()));

    const auto propagated = lio::propagateImu(samples, lio::ImuPropagationState{});

    ASSERT_TRUE(propagated.valid);
    EXPECT_EQ(propagated.used_samples, 2u);
    EXPECT_EQ(propagated.stamp.nsec, 1000000000LL);
}

}  // namespace test
}  // namespace n3mapping
