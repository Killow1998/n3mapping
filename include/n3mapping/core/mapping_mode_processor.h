// ROS-independent mapping-mode processor used by thin system wrappers.
#pragma once

#include <cstdint>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"

namespace n3mapping {
namespace core {

class MappingModeProcessor {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;

    struct Result {
        bool accepted_keyframe = false;
        int64_t keyframe_id = -1;
        Eigen::Isometry3d publish_pose = Eigen::Isometry3d::Identity();
        bool optimized_pose_available = false;
    };

    MappingModeProcessor(const Config& config,
                         KeyframeManager& keyframe_manager,
                         LoopDetector& loop_detector,
                         GraphOptimizer& graph_optimizer);

    Result process(double timestamp,
                   const Eigen::Isometry3d& pose_odom,
                   const PointCloud::Ptr& cloud,
                   const Eigen::Matrix<double, 6, 6>* covariance = nullptr);

private:
    const Config& config_;
    KeyframeManager& keyframe_manager_;
    LoopDetector& loop_detector_;
    GraphOptimizer& graph_optimizer_;
};

}  // namespace core
}  // namespace n3mapping
