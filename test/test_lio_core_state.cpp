#include <gtest/gtest.h>

#include "n3mapping/lio/core_state.h"

namespace n3mapping {
namespace test {

TEST(LioCoreStateTest, ConvertsImuPropagationToState) {
    lio::ImuPropagationState propagation;
    propagation.valid = true;
    propagation.stamp.nsec = 42;
    propagation.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    propagation.velocity = Eigen::Vector3d(4.0, 5.0, 6.0);
    propagation.orientation =
        Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ());

    const auto state = lio::stateFromImuPropagation(propagation);

    EXPECT_TRUE(state.initialized);
    EXPECT_EQ(state.stamp.nsec, 42);
    EXPECT_NEAR(state.T_world_lidar.translation().x(), 1.0, 1e-12);
    EXPECT_NEAR(state.velocity_world.z(), 6.0, 1e-12);
    EXPECT_NEAR(state.T_world_lidar.linear()(0, 0), std::cos(0.5), 1e-12);
}

TEST(LioCoreStateTest, ConvertsStateToFrame) {
    lio::LioCoreState state;
    state.initialized = true;
    state.stamp.nsec = 99;
    state.T_world_lidar.translation().x() = 1.5;

    const auto frame = lio::frameFromState(state);

    EXPECT_TRUE(frame.pose_valid);
    EXPECT_EQ(frame.stamp.nsec, 99);
    EXPECT_NEAR(frame.T_world_lidar.translation().x(), 1.5, 1e-12);
}

}  // namespace test
}  // namespace n3mapping
