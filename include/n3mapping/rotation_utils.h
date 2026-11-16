#pragma once

#include <cmath>
#include <Eigen/Geometry>

namespace n3mapping {

inline Eigen::Vector3d canonicalRollPitchYaw(const Eigen::Isometry3d& pose) {
    const auto rotation = pose.rotation();
    const double horizontal = std::hypot(rotation(0, 0), rotation(1, 0));
    const double pitch = std::atan2(-rotation(2, 0), horizontal);
    if (horizontal < 1e-10) {

        return {0.0, pitch, std::atan2(-rotation(0, 1), rotation(1, 1))};
    }
    return {std::atan2(rotation(2, 1), rotation(2, 2)), pitch,
            std::atan2(rotation(1, 0), rotation(0, 0))};
}

}
