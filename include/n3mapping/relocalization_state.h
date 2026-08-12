// ROS-free relocalization state and pose-source contracts.
#pragma once

#include <cmath>
#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace n3mapping {

enum class RelocalizationState : std::uint8_t {
  SEARCHING = 0,
  // Preserve the old wire-level meaning of value 1 while giving the state an
  // explicit non-authoritative contract.
  PROVISIONAL = 1,
  FULL_6DOF_LOCKED = 2,
  // Preserve the old wire-level odom-predicted meaning of value 3.
  RECENTLY_LOST = 3,
  DEGRADED_TRACKING = 4,
  LOST = 5,
};

enum class PoseSource : std::uint8_t {
  NONE = 0,
  ODOM_PREDICTED = 1,
  GEOMETRICALLY_CORRECTED = 2,
};

inline const char *relocalizationStateName(RelocalizationState state) {
  switch (state) {
  case RelocalizationState::SEARCHING:
    return "SEARCHING";
  case RelocalizationState::PROVISIONAL:
    return "PROVISIONAL";
  case RelocalizationState::FULL_6DOF_LOCKED:
    return "FULL_6DOF_LOCKED";
  case RelocalizationState::RECENTLY_LOST:
    return "RECENTLY_LOST";
  case RelocalizationState::DEGRADED_TRACKING:
    return "DEGRADED_TRACKING";
  case RelocalizationState::LOST:
    return "LOST";
  }
  return "SEARCHING";
}

inline const char *poseSourceName(PoseSource source) {
  switch (source) {
  case PoseSource::NONE:
    return "NONE";
  case PoseSource::ODOM_PREDICTED:
    return "ODOM_PREDICTED";
  case PoseSource::GEOMETRICALLY_CORRECTED:
    return "GEOMETRICALLY_CORRECTED";
  }
  return "NONE";
}

inline bool hasUsableGlobalRelocalizationPose(RelocalizationState state,
                                              PoseSource source) {
  return (state == RelocalizationState::FULL_6DOF_LOCKED &&
          source == PoseSource::GEOMETRICALLY_CORRECTED) ||
         (state == RelocalizationState::RECENTLY_LOST &&
          source == PoseSource::ODOM_PREDICTED);
}

inline bool hasAuthoritativeRelocalizationInitializationPose(
    RelocalizationState state, PoseSource source) {
  return state == RelocalizationState::FULL_6DOF_LOCKED &&
         source == PoseSource::GEOMETRICALLY_CORRECTED;
}

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

} // namespace n3mapping
