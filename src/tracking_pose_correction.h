#pragma once

#include <algorithm>
#include <Eigen/Geometry>

namespace n3mapping {

inline Eigen::Isometry3d trackingPoseCorrection(
    const Eigen::Isometry3d& current, const Eigen::Isometry3d& odom_pose,
    const Eigen::Isometry3d& endpoint, double confidence, bool strict,
    Eigen::Vector3d* remaining_shift = nullptr,
    const Eigen::Vector3d& odom_step = Eigen::Vector3d::Zero()) {
  const Eigen::Isometry3d target = endpoint * odom_pose.inverse();
  if (strict) {
    if (remaining_shift) remaining_shift->setZero();
    return target;
  }
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
  if (remaining_shift) {
    Eigen::Vector3d step = alpha * *remaining_shift;
    step.z() *= 0.3;
    if (step.dot(odom_step) < 0.0) {
      const double travel = odom_step.norm();
      const Eigen::Vector3d direction = odom_step / travel;
      const double opposing = -step.dot(direction);
      const double allowed = travel * opposing / (travel + opposing);
      const Eigen::Vector3d deferred = (opposing - allowed) * direction;
      filtered.translation() += deferred;
      step += deferred;
    }
    *remaining_shift -= step;
  }
  const Eigen::Quaterniond from(prediction.rotation()), to(endpoint.rotation());
  filtered.linear() =
      from.normalized().slerp(alpha, to.normalized()).normalized().toRotationMatrix();
  return filtered * odom_pose.inverse();
}
}
