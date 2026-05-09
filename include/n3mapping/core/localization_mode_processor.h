// ROS-independent localization-mode processor used by system wrappers.
#pragma once

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/world_localizing.h"

namespace n3mapping {
namespace core {

class LocalizationModeProcessor {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;

    struct Result {
        bool map_loaded = false;
        bool success = false;
        bool relocalization_locked = false;
        Eigen::Isometry3d publish_pose = Eigen::Isometry3d::Identity();
    };

    explicit LocalizationModeProcessor(WorldLocalizing& world_localizing);

    Result process(bool map_loaded,
                   const Eigen::Isometry3d& pose_odom,
                   const PointCloud::Ptr& cloud);

private:
    WorldLocalizing& world_localizing_;
};

}  // namespace core
}  // namespace n3mapping
