// ROS-independent map-extension mode processor used by system wrappers.
#pragma once

#include <cstdint>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/mapping_resuming.h"
#include "n3mapping/world_localizing.h"

namespace n3mapping {
namespace core {

class MapExtensionModeProcessor {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;

    struct Result {
        bool map_loaded = false;
        bool should_publish = false;
        bool initial_relocalization_attempted = false;
        bool initial_relocalization_success = false;
        bool accepted_keyframe = false;
        bool log_optimization = false;
        int64_t keyframe_id = -1;
        int cross_loops = 0;
        Eigen::Isometry3d publish_pose = Eigen::Isometry3d::Identity();
    };

    MapExtensionModeProcessor(KeyframeManager& keyframe_manager,
                              GraphOptimizer& graph_optimizer,
                              WorldLocalizing& world_localizing,
                              MappingResuming& mapping_resuming);

    Result process(bool map_loaded,
                   double timestamp,
                   const Eigen::Isometry3d& pose_odom,
                   const PointCloud::Ptr& cloud);

private:
    KeyframeManager& keyframe_manager_;
    GraphOptimizer& graph_optimizer_;
    WorldLocalizing& world_localizing_;
    MappingResuming& mapping_resuming_;
};

}  // namespace core
}  // namespace n3mapping
