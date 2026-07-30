// odometry_sanity.cpp
#include "n3mapping/odometry_sanity.h"

#include <cmath>

namespace n3mapping {

const OdometrySanityVerdict &
OdometrySanity::check(double timestamp, const Eigen::Isometry3d &pose) {
  if (verdict_.diverged) {
    return verdict_;
  }

  const bool pose_finite = pose.matrix().allFinite();
  if (!has_previous_) {
    if (pose_finite) {
      has_previous_ = true;
      previous_timestamp_ = timestamp;
      previous_pose_ = pose;
    }
    return verdict_;
  }

  const double dt = timestamp - previous_timestamp_;
  if (dt < limits_.min_dt_s) {
    // Out of order or duplicated stamps say nothing about the front end.
    return verdict_;
  }

  double speed = 0.0;
  double rate = 0.0;
  if (pose_finite) {
    speed = (pose.translation() - previous_pose_.translation()).norm() / dt;
    const Eigen::AngleAxisd delta(previous_pose_.rotation().transpose() *
                                  pose.rotation());
    rate = std::abs(delta.angle()) * 180.0 / M_PI / dt;
  }

  // A non-finite pose is already past any limit; treat it as a violation
  // rather than letting the comparisons below decide it silently.
  const bool violated = !pose_finite || speed > limits_.max_speed_mps ||
                        rate > limits_.max_angular_rate_dps;
  if (violated) {
    ++consecutive_;
    if (consecutive_ >= limits_.max_consecutive_violations) {
      verdict_.diverged = true;
      verdict_.speed_mps = speed;
      verdict_.angular_rate_dps = rate;
      verdict_.timestamp = timestamp;
      verdict_.consecutive = consecutive_;
    }
  } else {
    consecutive_ = 0;
  }

  if (pose_finite) {
    previous_timestamp_ = timestamp;
    previous_pose_ = pose;
  }
  return verdict_;
}

} // namespace n3mapping
