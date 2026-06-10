#include "n3mapping/global_map_cache.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <pcl/common/transforms.h>

#include "n3mapping/pcl_compat.h"

namespace n3mapping {

namespace {

template <typename T>
void hashCombine(std::size_t* seed, const T& value)
{
    *seed ^= std::hash<T>()(value) + 0x9e3779b97f4a7c15ULL + (*seed << 6) + (*seed >> 2);
}

bool finitePoint(const pcl::PointXYZI& point)
{
    return std::isfinite(static_cast<double>(point.x)) &&
           std::isfinite(static_cast<double>(point.y)) &&
           std::isfinite(static_cast<double>(point.z));
}

}  // namespace

std::size_t GlobalMapCache::VoxelKeyHash::operator()(const VoxelKey& key) const
{
    std::size_t seed = 0;
    hashCombine(&seed, key.x);
    hashCombine(&seed, key.y);
    hashCombine(&seed, key.z);
    return seed;
}

GlobalMapCache::GlobalMapCache(double voxel_size)
  : voxel_size_(std::isfinite(voxel_size) && voxel_size > 0.0 ? voxel_size : 0.0)
  , cloud_(pcl::make_shared<PointCloud>())
{
}

void GlobalMapCache::setVoxelSize(double voxel_size)
{
    const double normalized = std::isfinite(voxel_size) && voxel_size > 0.0 ? voxel_size : 0.0;
    if (normalized == voxel_size_) {
        return;
    }
    voxel_size_ = normalized;
    clear();
}

void GlobalMapCache::clear()
{
    resetStorage();
    full_rebuild_required_ = true;
    ++revision_;
}

void GlobalMapCache::markFullRebuildRequired()
{
    full_rebuild_required_ = true;
}

GlobalMapCache::PointCloudConstPtr GlobalMapCache::update(const std::vector<Keyframe::Ptr>& keyframes)
{
    if (full_rebuild_required_) {
        resetStorage();
    }

    std::vector<Keyframe::Ptr> ordered;
    ordered.reserve(keyframes.size());
    for (const auto& keyframe : keyframes) {
        if (keyframe) {
            ordered.push_back(keyframe);
        }
    }
    std::sort(ordered.begin(), ordered.end(), [](const Keyframe::Ptr& lhs, const Keyframe::Ptr& rhs) {
        return lhs->id < rhs->id;
    });

    bool changed = false;
    for (const auto& keyframe : ordered) {
        if (!keyframe || cached_keyframe_ids_.count(keyframe->id) != 0) {
            continue;
        }
        if (appendKeyframe(keyframe)) {
            cached_keyframe_ids_.insert(keyframe->id);
            changed = true;
        }
    }

    if (changed && useVoxelGrid()) {
        rebuildCloudFromVoxelCache();
    } else if (changed) {
        finalizeRawCloud();
    }

    if (changed || full_rebuild_required_) {
        ++revision_;
    }
    full_rebuild_required_ = false;
    return cloud_;
}

bool GlobalMapCache::useVoxelGrid() const
{
    return std::isfinite(voxel_size_) && voxel_size_ > 0.0;
}

void GlobalMapCache::resetStorage()
{
    cloud_ = pcl::make_shared<PointCloud>();
    cached_keyframe_ids_.clear();
    voxel_cache_.clear();
}

bool GlobalMapCache::appendKeyframe(const Keyframe::Ptr& keyframe)
{
    if (!keyframe || keyframe->id < 0 || !keyframe->cloud || keyframe->cloud->empty()) {
        return false;
    }

    PointCloud transformed;
    pcl::transformPointCloud(*keyframe->cloud, transformed, keyframe->pose_optimized.matrix().cast<float>());
    if (transformed.empty()) {
        return false;
    }

    bool appended_any = false;
    if (!useVoxelGrid()) {
        cloud_->points.reserve(cloud_->points.size() + transformed.points.size());
        for (const auto& point : transformed.points) {
            if (!finitePoint(point)) {
                continue;
            }
            cloud_->points.push_back(point);
            appended_any = true;
        }
        return appended_any;
    }

    for (const auto& point : transformed.points) {
        VoxelKey key;
        if (!voxelKeyForPoint(point, &key)) {
            continue;
        }
        auto& accum = voxel_cache_[key];
        accum.sx += point.x;
        accum.sy += point.y;
        accum.sz += point.z;
        accum.si += point.intensity;
        ++accum.count;
        appended_any = true;
    }
    return appended_any;
}

bool GlobalMapCache::voxelKeyForPoint(const pcl::PointXYZI& point, VoxelKey* key) const
{
    if (!key || !useVoxelGrid() || !finitePoint(point)) {
        return false;
    }

    const double inv_leaf = 1.0 / voxel_size_;
    const auto toCoord = [inv_leaf](float value, std::int64_t* out) {
        const double coord = std::floor(static_cast<double>(value) * inv_leaf);
        if (!std::isfinite(coord) ||
            coord < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
            coord > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
            return false;
        }
        *out = static_cast<std::int64_t>(coord);
        return true;
    };

    return toCoord(point.x, &key->x) &&
           toCoord(point.y, &key->y) &&
           toCoord(point.z, &key->z);
}

void GlobalMapCache::rebuildCloudFromVoxelCache()
{
    auto cloud = pcl::make_shared<PointCloud>();
    cloud->points.reserve(voxel_cache_.size());
    for (const auto& entry : voxel_cache_) {
        const auto& accum = entry.second;
        if (accum.count == 0) {
            continue;
        }
        const double inv_count = 1.0 / static_cast<double>(accum.count);
        pcl::PointXYZI point;
        point.x = static_cast<float>(accum.sx * inv_count);
        point.y = static_cast<float>(accum.sy * inv_count);
        point.z = static_cast<float>(accum.sz * inv_count);
        point.intensity = static_cast<float>(accum.si * inv_count);
        cloud->points.push_back(point);
    }
    cloud_ = cloud;
    finalizeRawCloud();
}

void GlobalMapCache::finalizeRawCloud()
{
    if (!cloud_) {
        cloud_ = pcl::make_shared<PointCloud>();
    }
    cloud_->width = static_cast<std::uint32_t>(cloud_->points.size());
    cloud_->height = 1;
    cloud_->is_dense = false;
}

}  // namespace n3mapping
