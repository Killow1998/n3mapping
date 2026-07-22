// ROS-free relocalization state and pose-source contracts.
#pragma once

namespace n3mapping {

enum class RelocalizationState {
  SEARCHING,
  REGION_HYPOTHESIS,
  FULL_6DOF_LOCKED,
  DEGRADED_TRACKING,
};

enum class PoseSource {
  NONE,
  ODOM_PREDICTED,
  GEOMETRICALLY_CORRECTED,
};

inline const char *relocalizationStateName(RelocalizationState state) {
  switch (state) {
  case RelocalizationState::SEARCHING:
    return "SEARCHING";
  case RelocalizationState::REGION_HYPOTHESIS:
    return "REGION_HYPOTHESIS";
  case RelocalizationState::FULL_6DOF_LOCKED:
    return "FULL_6DOF_LOCKED";
  case RelocalizationState::DEGRADED_TRACKING:
    return "DEGRADED_TRACKING";
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

} // namespace n3mapping
