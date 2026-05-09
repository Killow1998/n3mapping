#include "n3mapping/lio/local_map.h"

#include <algorithm>
#include <limits>

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

LioLocalMap::AlignmentStats LioLocalMap::estimateAlignmentCorrection(
    const LioCoreState& predicted_state,
    const PointCloud::ConstPtr& lidar_cloud,
    double max_correspondence_distance) const {
    AlignmentStats stats;
    if (!predicted_state.initialized || !lidar_cloud || lidar_cloud->empty() ||
        !aggregate_ || aggregate_->empty() || max_correspondence_distance <= 0.0) {
        return stats;
    }

    PointCloud source_world;
    pcl::transformPointCloud(*lidar_cloud, source_world,
                             predicted_state.T_world_lidar.matrix().cast<float>());
    stats.source_points = source_world.size();

    const double max_sq = max_correspondence_distance * max_correspondence_distance;
    Eigen::Vector3d source_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_sum = Eigen::Vector3d::Zero();
    double sq_error_sum = 0.0;

    for (const auto& source : source_world) {
        double best_sq = std::numeric_limits<double>::infinity();
        const pcl::PointXYZI* best_target = nullptr;
        for (const auto& target : *aggregate_) {
            const double dx = static_cast<double>(target.x) - source.x;
            const double dy = static_cast<double>(target.y) - source.y;
            const double dz = static_cast<double>(target.z) - source.z;
            const double sq = dx * dx + dy * dy + dz * dz;
            if (sq < best_sq) {
                best_sq = sq;
                best_target = &target;
            }
        }
        if (!best_target || best_sq > max_sq) {
            continue;
        }
        source_sum += Eigen::Vector3d(source.x, source.y, source.z);
        target_sum += Eigen::Vector3d(best_target->x, best_target->y, best_target->z);
        sq_error_sum += best_sq;
        ++stats.matched_points;
    }

    if (stats.matched_points == 0) {
        return stats;
    }
    const double inv = 1.0 / static_cast<double>(stats.matched_points);
    stats.centroid_correction_world = (target_sum - source_sum) * inv;
    stats.mean_squared_error = sq_error_sum * inv;
    stats.valid = true;
    return stats;
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
