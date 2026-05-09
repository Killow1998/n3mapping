// Small ROS-free IMU propagation utility for in-process LIO cores.
#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Geometry>

#include "n3mapping/core/types.h"

namespace n3mapping {
namespace lio {

struct ImuPropagationOptions {
    Eigen::Vector3d gravity_world = Eigen::Vector3d::Zero();
};

struct ImuPropagationState {
    core::TimeStamp stamp;
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    size_t used_samples = 0;
    bool valid = false;
};

ImuPropagationState propagateImu(const std::vector<core::ImuSample>& samples,
                                 const ImuPropagationState& initial_state,
                                 const ImuPropagationOptions& options = {});

}  // namespace lio
}  // namespace n3mapping
