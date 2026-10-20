// Pure candidate grouping and seed planning, independent of runtime state.
#include "relocalization_seed_plan.h"

#include <algorithm>
#include <limits>

namespace n3mapping {
RelocalizationSeedPlan
makeRelocalizationSeedPlan(const std::vector<LoopCandidate> &candidates,
                           KeyframeManager &keyframes, const Config &config) {
  RelocalizationSeedPlan plan;
  auto &basins = plan.basins;
  basins.reserve(kRelocMaxBasinCount);

  // Candidates are already sorted by coarse descriptor distance.
  // Build up to top-K basins in XY, then verify per-basin with local-submap
  // ICP.
  for (const auto &candidate : candidates) {
    auto kf = keyframes.getKeyframe(candidate.match_id);
    if (!kf)
      continue;
    const Eigen::Vector3d pos = kf->pose_optimized.translation();

    int best_basin = -1;
    double best_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < basins.size(); ++i) {
      const double d = (basins[i].center_pos.head<2>() - pos.head<2>()).norm();
      if (d < best_dist) {
        best_dist = d;
        best_basin = static_cast<int>(i);
      }
    }

    if (best_basin >= 0 && best_dist <= kRelocBasinAssignRadiusXY) {
      basins[best_basin].members.push_back(candidate);
    } else if (static_cast<int>(basins.size()) < kRelocMaxBasinCount) {
      RelocalizationSeedPlan::Basin bg;
      bg.center_match_id = candidate.match_id;
      bg.center_pos = pos;
      bg.members.push_back(candidate);
      basins.push_back(std::move(bg));
    }
  }
  auto &seeds = plan.seeds;
  for (const auto &basin : basins) {
    const auto count =
        std::min<std::size_t>(basin.members.size(), kRelocPerBasinVerifyCount);
    for (std::size_t i = 0; i < count; ++i) {
      for (const auto yaw :
           buildRelocalizationYawHypotheses(basin.members[i], config))
        seeds.emplace_back(basin.members[i].match_id, yaw);
    }
  }
  std::sort(seeds.begin(), seeds.end());
  return plan;
}

} // namespace n3mapping
