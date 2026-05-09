#include "n3mapping/lio/dlio_map_accumulator.h"

#include <algorithm>

#include <pcl/filters/voxel_grid.h>

namespace n3mapping {
namespace lio {
namespace dlio {

MapAccumulator::MapAccumulator()
    : MapAccumulator(Options{}) {}

MapAccumulator::MapAccumulator(const Options& options)
    : options_(options),
      map_(pcl::make_shared<PointCloud>()) {
    options_.dense_input_skip = std::max<size_t>(1, options_.dense_input_skip);
    map_->height = 1;
    map_->is_dense = true;
}

MapAccumulator::AddResult MapAccumulator::addKeyframe(
    const PointCloud::ConstPtr& keyframe) {
    AddResult result;
    ++inputs_seen_;
    ++skip_counter_;
    if (skip_counter_ < options_.dense_input_skip) {
        result.map_points = map_->size();
        return result;
    }
    skip_counter_ = 0;

    if (!keyframe || keyframe->empty()) {
        result.map_points = map_->size();
        return result;
    }

    result.input_points = keyframe->size();
    const auto filtered = filteredCopy(keyframe);
    result.filtered_points = filtered->size();
    if (filtered->empty()) {
        result.map_points = map_->size();
        return result;
    }

    *map_ += *filtered;
    map_->width = static_cast<uint32_t>(map_->size());
    map_->height = 1;
    map_->is_dense = map_->is_dense && filtered->is_dense;
    ++accepted_keyframes_;

    result.accepted = true;
    result.map_points = map_->size();
    return result;
}

void MapAccumulator::clear() {
    inputs_seen_ = 0;
    accepted_keyframes_ = 0;
    skip_counter_ = 0;
    map_->clear();
    map_->width = 0;
    map_->height = 1;
    map_->is_dense = true;
}

MapAccumulator::PointCloud::Ptr MapAccumulator::filteredCopy(
    const PointCloud::ConstPtr& keyframe) const {
    auto filtered = pcl::make_shared<PointCloud>();
    if (options_.leaf_size <= 0.0) {
        *filtered = *keyframe;
        filtered->width = static_cast<uint32_t>(filtered->size());
        filtered->height = 1;
        return filtered;
    }

    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    const float leaf = static_cast<float>(options_.leaf_size);
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.setInputCloud(keyframe);
    voxel.filter(*filtered);
    filtered->width = static_cast<uint32_t>(filtered->size());
    filtered->height = 1;
    return filtered;
}

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
