#pragma once

#include <cmath>
#include <Eigen/Geometry>

namespace n3mapping {

inline bool isFiniteRigidPose(const Eigen::Isometry3d &pose,
                              double tolerance = 1e-6) {
  if (!pose.matrix().allFinite()) {
    return false;
  }
  const Eigen::Matrix3d rotation = pose.linear();
  return (rotation.transpose() * rotation)
             .isApprox(Eigen::Matrix3d::Identity(), tolerance) &&
         std::abs(rotation.determinant() - 1.0) <= tolerance &&
         pose.matrix().row(3).isApprox(
             Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), tolerance);
}

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
