#include "n3mapping/core/map_extension_mode_processor.h"

#include <exception>

#include <glog/logging.h>

namespace n3mapping {
namespace core {

MapExtensionModeProcessor::MapExtensionModeProcessor(
    KeyframeManager& keyframe_manager,
    GraphOptimizer& graph_optimizer,
    WorldLocalizing& world_localizing,
    MappingResuming& mapping_resuming)
    : keyframe_manager_(keyframe_manager)
    , graph_optimizer_(graph_optimizer)
    , world_localizing_(world_localizing)
    , mapping_resuming_(mapping_resuming) {}

MapExtensionModeProcessor::Result MapExtensionModeProcessor::process(
    bool map_loaded,
    double timestamp,
    const Eigen::Isometry3d& pose_odom,
    const PointCloud::Ptr& cloud) {
    Result result;
    result.map_loaded = map_loaded;
    result.publish_pose = pose_odom;

    if (!map_loaded) {
        return result;
    }

    auto state = mapping_resuming_.getState();
    if (state == MappingResumingState::MAP_LOADED) {
        result.initial_relocalization_attempted = true;
        result.initial_relocalization_success =
            mapping_resuming_.performInitialRelocalization(cloud, pose_odom);
        return result;
    }

    if (state != MappingResumingState::RELOCALIZED &&
        state != MappingResumingState::EXTENDING) {
        return result;
    }

    auto T_map_odom = world_localizing_.getMapToOdomTransform();
    Eigen::Isometry3d pose_map = T_map_odom * pose_odom;

    if (!keyframe_manager_.shouldAddKeyframe(pose_odom)) {
        result.should_publish = true;
        result.publish_pose = pose_map;
        return result;
    }

    int64_t kf_id = mapping_resuming_.processNewKeyframe(timestamp, pose_odom, cloud);
    if (kf_id >= 0) {
        result.accepted_keyframe = true;
        result.keyframe_id = kf_id;
        result.cross_loops = mapping_resuming_.detectCrossLoops(kf_id);

        graph_optimizer_.incrementalOptimize();

        auto optimized_poses = graph_optimizer_.getOptimizedPoses();
        keyframe_manager_.updateOptimizedPoses(optimized_poses);

        if (graph_optimizer_.hasNode(kf_id)) {
            try {
                pose_map = graph_optimizer_.getOptimizedPose(kf_id);
            } catch (const std::exception& e) {
                LOG(WARNING) << "Failed to get optimized pose: " << e.what();
            }
        }
        result.log_optimization = true;
    }

    result.should_publish = true;
    result.publish_pose = pose_map;
    return result;
}

}  // namespace core
}  // namespace n3mapping
