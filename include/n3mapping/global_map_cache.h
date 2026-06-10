#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/keyframe.h"

namespace n3mapping {

class GlobalMapCache {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;
    using PointCloudPtr = PointCloud::Ptr;
    using PointCloudConstPtr = PointCloud::ConstPtr;

    explicit GlobalMapCache(double voxel_size = 0.0);

    void setVoxelSize(double voxel_size);
    double voxelSize() const { return voxel_size_; }

    void clear();
    void markFullRebuildRequired();

    PointCloudConstPtr update(const std::vector<Keyframe::Ptr>& keyframes);
    PointCloudConstPtr cloud() const { return cloud_; }

    std::uint64_t revision() const { return revision_; }
    std::size_t cachedKeyframeCount() const { return cached_keyframe_ids_.size(); }
    bool fullRebuildRequired() const { return full_rebuild_required_; }

private:
    struct VoxelKey {
        std::int64_t x = 0;
        std::int64_t y = 0;
        std::int64_t z = 0;

        bool operator==(const VoxelKey& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash {
        std::size_t operator()(const VoxelKey& key) const;
    };

    struct VoxelAccum {
        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
        double si = 0.0;
        std::uint32_t count = 0;
    };

    bool useVoxelGrid() const;
    void resetStorage();
    bool appendKeyframe(const Keyframe::Ptr& keyframe);
    bool voxelKeyForPoint(const pcl::PointXYZI& point, VoxelKey* key) const;
    void rebuildCloudFromVoxelCache();
    void finalizeRawCloud();

    double voxel_size_ = 0.0;
    bool full_rebuild_required_ = true;
    std::uint64_t revision_ = 0;
    PointCloudPtr cloud_;
    std::unordered_set<int64_t> cached_keyframe_ids_;
    std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> voxel_cache_;
};

}  // namespace n3mapping
