#pragma once

#include <algorithm>
#include <Eigen/Geometry>

namespace n3mapping {
// Compute the correction once, with identical semantics for admission and
// publication. Ordinary localization smooths map-to-odometry, not map poses.
inline Eigen::Isometry3d trackingPoseCorrection(
    const Eigen::Isometry3d& current, const Eigen::Isometry3d& odom_pose,
    const Eigen::Isometry3d& endpoint, double confidence, bool strict) {
  const Eigen::Isometry3d target = endpoint * odom_pose.inverse();
  if (strict) return target;
  const double distance =
      ((current * odom_pose).inverse() * endpoint).translation().norm();
  const double alpha = std::max(
      distance <= 0.5 ? std::min(confidence * 0.3, 0.2)
                     : distance <= 2.0 ? std::min(confidence * 0.15, 0.1)
                                       : std::min(confidence * 0.08, 0.05),
      0.01);
  Eigen::Isometry3d correction = current;
  correction.translation() += alpha * (target.translation() - current.translation());
  correction.translation().z() = current.translation().z() +
      alpha * 0.3 * (target.translation().z() - current.translation().z());
  const Eigen::Quaterniond from(current.rotation()), to(target.rotation());
  correction.linear() =
      from.normalized().slerp(alpha, to.normalized()).normalized().toRotationMatrix();
  return correction;
}
} // namespace n3mapping
