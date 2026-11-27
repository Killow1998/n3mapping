#include "n3mapping/map_serializer.h"
#include "n3mapping/global_map_cache.h"
#include "n3mapping/keyframe_manager.h"
#include <filesystem>
#include <pcl/io/pcd_io.h>

namespace n3mapping {

MapSerializer::MapSerializer(const Config& config) : config_(config) {}

bool MapSerializer::saveGlobalMap(const std::string& filepath,
                                  const pcl::PointCloud<pcl::PointXYZI>& cloud) {
    if (cloud.empty()) return false;
    try {
        const std::filesystem::path path(filepath);
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        return pcl::io::savePCDFileBinary(filepath, cloud) != -1;
    } catch (const std::exception&) { return false; }
}

bool MapSerializer::saveGlobalMap(const std::string& filepath,
                                  const KeyframeManager& keyframe_manager,
                                  double voxel_size) {
    return saveGlobalMap(filepath, *buildGlobalMap(keyframe_manager, voxel_size));
}

pcl::PointCloud<pcl::PointXYZI>::Ptr MapSerializer::buildGlobalMap(const KeyframeManager& keyframe_manager,
                                                                   double voxel_size) const {
    return GlobalMapCache::build(keyframe_manager.getAllKeyframes(), voxel_size);
}

}
