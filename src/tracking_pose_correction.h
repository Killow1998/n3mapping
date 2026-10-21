#pragma once

#include <algorithm>
#include <Eigen/Geometry>

namespace n3mapping {
// Compute the correction once, with identical semantics for admission and
// publication. Blend at the observed body pose, then recover map-to-odometry;
// the result must not depend on an arbitrary shift of the odometry origin.
inline Eigen::Isometry3d trackingPoseCorrection(
    const Eigen::Isometry3d& current, const Eigen::Isometry3d& odom_pose,
    const Eigen::Isometry3d& endpoint, double confidence, bool strict) {
  const Eigen::Isometry3d target = endpoint * odom_pose.inverse();
  if (strict) return target;
  const Eigen::Isometry3d prediction = current * odom_pose;
  const double distance = (prediction.inverse() * endpoint).translation().norm();
  const double alpha = std::max(
      distance <= 0.5 ? std::min(confidence * 0.3, 0.2)
                     : distance <= 2.0 ? std::min(confidence * 0.15, 0.1)
                                       : std::min(confidence * 0.08, 0.05),
      0.01);
  Eigen::Isometry3d filtered = prediction;
  filtered.translation() += alpha * (endpoint.translation() - prediction.translation());
  filtered.translation().z() = prediction.translation().z() +
      alpha * 0.3 * (endpoint.translation().z() - prediction.translation().z());
  const Eigen::Quaterniond from(prediction.rotation()), to(endpoint.rotation());
  filtered.linear() =
      from.normalized().slerp(alpha, to.normalized()).normalized().toRotationMatrix();
  return filtered * odom_pose.inverse();
}
} // namespace n3mapping
