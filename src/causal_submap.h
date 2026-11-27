#pragma once
#include "n3mapping/keyframe_manager.h"
#include <map>
namespace n3mapping {

Keyframe::PointCloudT::Ptr
buildCausalSubmap(const std::map<int64_t, Keyframe::Ptr> &keyframes,
                  int64_t center_id, int range, int64_t root_id,
                  KeyframeManager::PoseSource poses);
Keyframe::PointCloudT::Ptr
maybeVoxelizeRhpdCloud(const Keyframe::PointCloudT::Ptr &cloud,
                       double leaf_size);
}
