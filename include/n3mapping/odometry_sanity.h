// odometry_sanity.h - refuses to keep mapping from odometry that has run away.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace n3mapping {

// FAST_LIO has no divergence detection of its own. On the f7-to-f9 recording it
// lost attitude lock inside a stairwell -- a shaft about 4 m across, where the
// median LiDAR range collapses from 3.6 m to 2.0 m and scan matching stops
// constraining the vertical -- and from then on gravity leaked into the
// horizontal axes as a residual of about 1.24 g. The velocity state integrated
// that open-loop for the remaining twelve minutes, reaching 8852 m/s, and the
// mapper consumed all of it: 6845 keyframes and a 3,108,188 m trajectory saved
// without a single complaint. Once the predicted pose is kilometres from the
// map there are no correspondences left, so the front end can never recover on
// its own and every frame after the runaway is worthless.
//
// The check is two subtractions, a norm and a compare per frame. It exists to
// catch the physically impossible, not to judge quality: the limits sit far
// above anything the platform can do, so a trip means the input is broken
// rather than merely poor.
struct OdometrySanityLimits {
  // A Go2-W tops out near 3 m/s; the b22 recording peaks at 2.66. Ten leaves
  // almost four times that headroom and still catches f7tof9, whose first
  // excursion reaches 11 m/s.
  double max_speed_mps = 10.0;
  // Two full revolutions per second. Foot impacts on a quadruped produce brief
  // high rates, so this is deliberately far above them.
  double max_angular_rate_dps = 720.0;
  // One outlier is a glitch; a run of them is a runaway. The f7tof9 divergence
  // is monotonic and never comes back, so it trips this within half a second.
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
