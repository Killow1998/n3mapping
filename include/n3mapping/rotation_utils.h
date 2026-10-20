#pragma once

#include <cmath>
#include <Eigen/Geometry>

namespace n3mapping {

// R = Rz(yaw) Ry(pitch) Rx(roll), with pitch in [-pi/2, pi/2].
// Eigen's XYZ Euler branch can turn a tiny negative roll into three angles
// near pi. Those equivalent coordinates must not become a yaw-flip decision.
inline Eigen::Vector3d canonicalRollPitchYaw(const Eigen::Isometry3d& pose) {
    const auto rotation = pose.rotation();
    const double horizontal = std::hypot(rotation(0, 0), rotation(1, 0));
    const double pitch = std::atan2(-rotation(2, 0), horizontal);
    if (horizontal < 1e-10) {
        // At the pitch singularity only the combined roll/yaw is observable.
        return {0.0, pitch, std::atan2(-rotation(0, 1), rotation(1, 1))};
    }
    return {std::atan2(rotation(2, 1), rotation(2, 2)), pitch,
            std::atan2(rotation(1, 0), rotation(0, 0))};
}

} // namespace n3mapping
