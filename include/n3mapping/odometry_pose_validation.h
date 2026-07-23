#pragma once

#include <cmath>

#include <Eigen/Geometry>

namespace n3mapping {

inline bool tryMakeRigidOdometryPose(
    double tx, double ty, double tz,
    double qx, double qy, double qz, double qw,
    Eigen::Isometry3d* pose) {
  if (pose == nullptr) {
    return false;
  }
  *pose = Eigen::Isometry3d::Identity();
  if (!std::isfinite(tx) || !std::isfinite(ty) || !std::isfinite(tz) ||
      !std::isfinite(qx) || !std::isfinite(qy) || !std::isfinite(qz) ||
      !std::isfinite(qw)) {
    return false;
  }
  Eigen::Quaterniond quaternion(qw, qx, qy, qz);
  constexpr double kMinimumSquaredNorm = 1e-12;
  if (!std::isfinite(quaternion.squaredNorm()) ||
      quaternion.squaredNorm() <= kMinimumSquaredNorm) {
    return false;
  }
  quaternion.normalize();
  pose->translation() << tx, ty, tz;
  pose->linear() = quaternion.toRotationMatrix();
  return pose->matrix().allFinite();
}

}  // namespace n3mapping
