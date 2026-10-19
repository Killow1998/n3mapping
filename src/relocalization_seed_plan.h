// The concrete descriptor seeds requested for one global search window.
#pragma once
#include "n3mapping/loop_detector.h"
#include "n3mapping/relocalization_target_provider.h"
#include <utility>
#include <vector>

namespace n3mapping {
inline constexpr int kRelocMaxBasinCount = 3;
inline constexpr int kRelocPerBasinVerifyCount = 3;
inline constexpr double kRelocBasinAssignRadiusXY = 4.0;

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
