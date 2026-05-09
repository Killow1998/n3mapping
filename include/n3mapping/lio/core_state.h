// Shared ROS-free state for in-process LIO cores.
#pragma once

#include <optional>

#include <Eigen/Geometry>

#include "n3mapping/core/types.h"
#include "n3mapping/lio/imu_propagator.h"

namespace n3mapping {
namespace lio {

struct LioCoreState {
    core::TimeStamp stamp;
    Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();
    Eigen::Vector3d velocity_world = Eigen::Vector3d::Zero();
    bool initialized = false;
};

LioCoreState stateFromImuPropagation(const ImuPropagationState& propagation);
core::LioFrame frameFromState(const LioCoreState& state);

}  // namespace lio
}  // namespace n3mapping
