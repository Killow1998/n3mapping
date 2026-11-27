#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "n3mapping/keyframe.h"

namespace n3mapping {

struct Config;

struct ContinuationCleaning {
  std::map<int64_t, Keyframe::PointCloudT::Ptr> clouds;
  std::size_t inspected_keyframes = 0;
  std::size_t skipped_empty_keyframes = 0;
  std::size_t removed_points = 0;
  std::size_t kept_points = 0;
};

ContinuationCleaning computeContinuationCleaning(
    const Config& config, const std::vector<Keyframe::Ptr>& keyframes);

}
