#include "n3mapping/map_serializer.h"
#include "n3mapping/global_map_cache.h"
#include "n3mapping/keyframe_manager.h"
#include <filesystem>
#include <pcl/io/pcd_io.h>

namespace n3mapping {

MapSerializer::MapSerializer(const Config& config) : config_(config) {}

bool MapSerializer::saveGlobalMap(const std::string& filepath, const KeyframeManager& keyframe_manager, double voxel_size) {
    try {
        std::filesystem::path fp(filepath);
        if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path());
        auto global = buildGlobalMap(keyframe_manager, voxel_size);
        if (global->empty()) return false;
        return pcl::io::savePCDFileBinary(filepath, *global) != -1;
    } catch (...) { return false; }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr MapSerializer::buildGlobalMap(const KeyframeManager& keyframe_manager,
                                                                   double voxel_size) const {
    return GlobalMapCache::build(keyframe_manager.getAllKeyframes(), voxel_size);
}

}
