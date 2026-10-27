// odometry_sanity.h - refuses to keep mapping from odometry that has run away.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace n3mapping {

// Reject sustained kinematically implausible odometry, not small accuracy errors.
// Limits exceed normal platform motion; consecutive violations separate runaway
// from isolated outliers.
struct OdometrySanityLimits {
  double max_speed_mps = 10.0;
  double max_angular_rate_dps = 720.0;
  int max_consecutive_violations = 5;
  // Below this the interval is too short to divide by.
  double min_dt_s = 1e-3;
};

struct OdometrySanityVerdict {
  bool diverged = false;
  // Populated on the frame that trips the check.
  double speed_mps = 0.0;
  double angular_rate_dps = 0.0;
  double timestamp = 0.0;
  // Current violation run, also before divergence is latched. Mapping must
  // not commit known-invalid samples while waiting to classify a runaway.
  int consecutive = 0;
};

// Feeds on consecutive front-end poses. Once it declares divergence it stays
// declared: there is no evidence that could show the front end recovered, since
// the thing that would supply it is the front end.
class OdometrySanity {
public:
  explicit OdometrySanity(const OdometrySanityLimits &limits = {})
      : limits_(limits) {}

  // Returns the running verdict. Call once per front-end frame, in order.
  const OdometrySanityVerdict &check(double timestamp,
                                     const Eigen::Isometry3d &pose);

  const OdometrySanityVerdict &verdict() const { return verdict_; }
  bool diverged() const { return verdict_.diverged; }

private:
  OdometrySanityLimits limits_;
  OdometrySanityVerdict verdict_;
  bool has_previous_ = false;
  double previous_timestamp_ = 0.0;
  Eigen::Isometry3d previous_pose_ = Eigen::Isometry3d::Identity();
  int consecutive_ = 0;
};

} // namespace n3mapping
