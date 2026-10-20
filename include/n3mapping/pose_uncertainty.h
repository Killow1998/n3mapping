#pragma once

#include <Eigen/Geometry>

namespace n3mapping {

// Information uses right SE(3) perturbations in (translation, rotation) order.
// For Z = C * P, delta_C = Ad(P) delta_Z, hence H_Z = Ad(P)^T H_C Ad(P).
inline Eigen::Matrix<double, 6, 6> rightComposedInformation(
    const Eigen::Matrix<double, 6, 6>& information,
    const Eigen::Isometry3d& right_transform) {
    const auto& p = right_transform.translation();
    Eigen::Matrix3d skew;
    skew << 0.0, -p.z(), p.y(), p.z(), 0.0, -p.x(), -p.y(), p.x(), 0.0;
    Eigen::Matrix<double, 6, 6> adjoint =
        Eigen::Matrix<double, 6, 6>::Zero();
    adjoint.topLeftCorner<3, 3>() = right_transform.rotation();
    adjoint.topRightCorner<3, 3>() = skew * right_transform.rotation();
    adjoint.bottomRightCorner<3, 3>() = right_transform.rotation();
    const Eigen::Matrix<double, 6, 6> transported =
        adjoint.transpose() * information * adjoint;
    return 0.5 * (transported + transported.transpose());
}

} // namespace n3mapping
