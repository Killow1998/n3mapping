// The concrete descriptor seeds requested for one global search window.
#pragma once
#include "n3mapping/loop_detector.h"
#include "n3mapping/relocalization_target_provider.h"
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace n3mapping {
inline constexpr int kRelocMaxBasinCount = 3;
inline constexpr int kRelocPerBasinVerifyCount = 3;
inline constexpr double kRelocBasinAssignRadiusXY = 4.0;

inline std::vector<double> buildRelocalizationYawHypotheses(
    const LoopCandidate& candidate, const Config& config) {
  auto yaws = buildDescriptorYawHypotheses(candidate, config);
  if (!candidate.fromRHPD() || !candidate.fromSC() ||
      !config.rhpd_use_sc_yaw || !std::isfinite(candidate.sc_distance) ||
      config.rhpd_yaw_hypotheses < 4) return yaws;
  // The descriptor-phased grid already covers the full circle, including the
  // opposite mode. Keep adjacent-sector starts for heading-only/reduced grids;
  // do not add two more local solves to every full-coverage place candidate.
  const double phase = -static_cast<double>(candidate.yaw_diff_rad);
  yaws.erase(std::remove_if(yaws.begin(), yaws.end(), [&](double yaw) {
    for (int i = 0; i < config.rhpd_yaw_hypotheses; ++i) {
      const double grid = phase + 2.0 * M_PI * i / config.rhpd_yaw_hypotheses;
      if (std::abs(std::remainder(yaw - grid, 2.0 * M_PI)) <
          kDescriptorYawMergeTolerance) return false;
    }
    return true;
  }), yaws.end());
  return yaws;
}

struct RelocalizationSeedPlan {
  struct Basin {
    int64_t center_match_id = -1;
    Eigen::Vector3d center_pos = Eigen::Vector3d::Zero();
    std::vector<LoopCandidate> members;
    std::vector<RelocTargetRequest> targets;
  };
  std::vector<Basin> basins;
  std::vector<std::pair<int64_t, double>> seeds;
};
RelocalizationSeedPlan
makeRelocalizationSeedPlan(const std::vector<LoopCandidate> &candidates,
                           KeyframeManager &keyframes, const Config &config);
} // namespace n3mapping
