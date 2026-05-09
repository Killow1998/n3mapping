#include "n3mapping/lio/local_map.h"

#include <algorithm>

#include <pcl/common/transforms.h>

namespace n3mapping {
namespace lio {

LioLocalMap::LioLocalMap(size_t max_keyframes)
    : max_keyframes_(std::max<size_t>(1, max_keyframes)),
      aggregate_(pcl::make_shared<PointCloud>()) {}

void LioLocalMap::clear() {
    keyframes_.clear();
    aggregate_->clear();
    aggregate_->width = 0;
    aggregate_->height = 1;
    aggregate_->is_dense = true;
}

bool LioLocalMap::addFrame(const LioCoreState& state,
                           const PointCloud::ConstPtr& lidar_cloud) {
    if (!state.initialized || !lidar_cloud || lidar_cloud->empty()) {
        return false;
    }

    auto cloud_world = pcl::make_shared<PointCloud>();
    pcl::transformPointCloud(*lidar_cloud, *cloud_world,
                             state.T_world_lidar.matrix().cast<float>());
    keyframes_.push_back(Keyframe{state, cloud_world});
    while (keyframes_.size() > max_keyframes_) {
        keyframes_.pop_front();
    }
    rebuildAggregate();
    return true;
}

void LioLocalMap::rebuildAggregate() {
    aggregate_->clear();
    aggregate_->is_dense = true;
    for (const auto& keyframe : keyframes_) {
        *aggregate_ += *keyframe.cloud_world;
        aggregate_->is_dense = aggregate_->is_dense && keyframe.cloud_world->is_dense;
    }
    aggregate_->width = static_cast<uint32_t>(aggregate_->size());
    aggregate_->height = 1;
}

}  // namespace lio
}  // namespace n3mapping
