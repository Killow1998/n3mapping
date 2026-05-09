#include <gtest/gtest.h>

#include <cmath>

#include "n3mapping/lio/dlio_imu_integration.h"

namespace n3mapping {
namespace test {
namespace {

lio::dlio::ImuMeasurement makeImu(double stamp,
                                  const Eigen::Vector3f& accel,
                                  const Eigen::Vector3f& gyro =
                                      Eigen::Vector3f::Zero()) {
    lio::dlio::ImuMeasurement imu;
    imu.stamp = stamp;
    imu.dt = stamp;
    imu.linear_acceleration = accel;
    imu.angular_velocity = gyro;
    return imu;
}

std::vector<lio::dlio::ImuMeasurement> makeSamples(
    const std::vector<double>& stamps,
    const Eigen::Vector3f& accel,
    const Eigen::Vector3f& gyro = Eigen::Vector3f::Zero()) {
    std::vector<lio::dlio::ImuMeasurement> samples;
    samples.reserve(stamps.size());
    double previous = 0.0;
    for (double stamp : stamps) {
        auto sample = makeImu(stamp, accel, gyro);
        sample.dt = samples.empty() ? stamp : stamp - previous;
        previous = stamp;
        samples.push_back(sample);
    }
    return samples;
}

}  // namespace

TEST(DlioImuIntegrationTest, RejectsInvalidCoverage) {
    lio::dlio::ImuIntegrationRequest request;
    request.start_time = 1.0;
    request.sorted_timestamps = {1.5};

    EXPECT_TRUE(lio::dlio::integrateImu({}, request).empty());
    EXPECT_TRUE(lio::dlio::integrateImu(
                    makeSamples({1.1, 1.2}, Eigen::Vector3f::Zero()),
                    request)
                    .empty());
}

TEST(DlioImuIntegrationTest, IntegratesConstantWorldAcceleration) {
    lio::dlio::ImuIntegrationRequest request;
    request.start_time = 0.0;
    request.gravity = 0.0;
    request.sorted_timestamps = {0.5, 1.0};
    const auto samples =
        makeSamples({0.0, 0.5, 1.0},
                    Eigen::Vector3f(1.0f, 0.0f, 0.0f));

    const auto poses = lio::dlio::integrateImu(samples, request);
    const auto states = lio::dlio::integrateImuStates(samples, request);

    ASSERT_EQ(poses.size(), 2u);
    ASSERT_EQ(states.size(), 2u);
    EXPECT_NEAR(poses[0](0, 3), 0.125f, 1e-5f);
    EXPECT_NEAR(poses[1](0, 3), 0.5f, 1e-5f);
    EXPECT_NEAR(states[0].velocity.x(), 0.5f, 1e-5f);
    EXPECT_NEAR(states[1].velocity.x(), 1.0f, 1e-5f);
    EXPECT_NEAR(states[1].stamp, 1.0, 1e-12);
    EXPECT_NEAR(poses[1](1, 3), 0.0f, 1e-6f);
}

TEST(DlioImuIntegrationTest, PreservesStationaryGravityCompensatedPose) {
    lio::dlio::ImuIntegrationRequest request;
    request.start_time = 0.0;
    request.gravity = 9.80665;
    request.sorted_timestamps = {1.0};
    const auto samples =
        makeSamples({0.0, 0.5, 1.0},
                    Eigen::Vector3f(0.0f, 0.0f, 9.80665f));

    const auto poses = lio::dlio::integrateImu(samples, request);

    ASSERT_EQ(poses.size(), 1u);
    EXPECT_NEAR(poses[0](0, 3), 0.0f, 1e-6f);
    EXPECT_NEAR(poses[0](1, 3), 0.0f, 1e-6f);
    EXPECT_NEAR(poses[0](2, 3), 0.0f, 1e-5f);
}

TEST(DlioImuIntegrationTest, IntegratesConstantYawRate) {
    lio::dlio::ImuIntegrationRequest request;
    request.start_time = 0.0;
    request.gravity = 0.0;
    request.sorted_timestamps = {1.0};
    const auto samples =
        makeSamples({0.0, 0.5, 1.0},
                    Eigen::Vector3f::Zero(),
                    Eigen::Vector3f(0.0f, 0.0f, 0.1f));

    const auto poses = lio::dlio::integrateImu(samples, request);

    ASSERT_EQ(poses.size(), 1u);
    const Eigen::Matrix2f rot = poses[0].block<2, 2>(0, 0);
    EXPECT_NEAR(rot(0, 0), std::cos(0.1f), 2e-4f);
    EXPECT_NEAR(rot(1, 0), std::sin(0.1f), 2e-4f);
}

}  // namespace test
}  // namespace n3mapping
