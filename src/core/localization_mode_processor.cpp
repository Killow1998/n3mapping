#include "n3mapping/core/localization_mode_processor.h"

namespace n3mapping {
namespace core {

LocalizationModeProcessor::LocalizationModeProcessor(WorldLocalizing& world_localizing)
    : world_localizing_(world_localizing) {}

LocalizationModeProcessor::Result LocalizationModeProcessor::process(
    bool map_loaded,
    const Eigen::Isometry3d& pose_odom,
    const PointCloud::Ptr& cloud) {
    Result result;
    result.map_loaded = map_loaded;
    result.publish_pose = pose_odom;

    if (!map_loaded) {
        return result;
    }

    bool success = false;
    bool relocalization_locked = false;
    Eigen::Isometry3d pose_map = Eigen::Isometry3d::Identity();

    if (world_localizing_.isRelocalized()) {
        auto track_result = world_localizing_.trackLocalization(cloud, pose_odom);
        if (track_result.success) {
            pose_map = track_result.pose_in_map;
            success = true;
        }
    }
    if (!world_localizing_.isRelocalized() || !success) {
        auto reloc_result = world_localizing_.relocalize(cloud, pose_odom);
        if (reloc_result.success) {
            pose_map = reloc_result.pose_in_map;
            success = true;
            relocalization_locked = true;
        }
    }

    result.success = success;
    result.relocalization_locked = relocalization_locked;
    result.publish_pose =
        success ? pose_map : world_localizing_.getMapToOdomTransform() * pose_odom;
    return result;
}

}  // namespace core
}  // namespace n3mapping
