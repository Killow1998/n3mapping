#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace n3mapping {

struct OdometrySanityLimits {
  double max_speed_mps = 10.0;
  double max_angular_rate_dps = 720.0;
  int max_consecutive_violations = 5;

  double min_dt_s = 1e-3;
};

struct OdometrySanityVerdict {
  bool diverged = false;

  double speed_mps = 0.0;
  double angular_rate_dps = 0.0;
  double timestamp = 0.0;

  int consecutive = 0;
};

class OdometrySanity {
public:
  explicit OdometrySanity(const OdometrySanityLimits &limits = {})
      : limits_(limits) {}

  const OdometrySanityVerdict &check(double timestamp,
                                     const Eigen::Isometry3d &pose);

  void reset() { *this = OdometrySanity(limits_); }

  const OdometrySanityVerdict &verdict() const { return verdict_; }
  bool diverged() const { return verdict_.diverged; }

private:
  OdometrySanityLimits limits_;
  OdometrySanityVerdict verdict_;
  bool has_previous_ = false;
  double previous_timestamp_ = 0.0;
  Eigen::Isometry3d previous_pose_ = Eigen::Isometry3d::Identity();
};

}
