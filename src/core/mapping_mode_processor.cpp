#include "n3mapping/core/mapping_mode_processor.h"

#include <algorithm>
#include <exception>

#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/memory.h>

namespace n3mapping {
namespace core {

MappingModeProcessor::MappingModeProcessor(const Config& config,
                                           KeyframeManager& keyframe_manager,
                                           LoopDetector& loop_detector,
                                           GraphOptimizer& graph_optimizer)
    : config_(config)
    , keyframe_manager_(keyframe_manager)
    , loop_detector_(loop_detector)
    , graph_optimizer_(graph_optimizer) {}

MappingModeProcessor::Result MappingModeProcessor::process(
    double timestamp,
    const Eigen::Isometry3d& pose_odom,
    const PointCloud::Ptr& cloud) {
    Result result;
    result.publish_pose = pose_odom;

    if (!keyframe_manager_.shouldAddKeyframe(pose_odom)) {
        return result;
    }

    const int64_t kf_id = keyframe_manager_.addKeyframe(timestamp, pose_odom, cloud);
    result.accepted_keyframe = true;
    result.keyframe_id = kf_id;

    loop_detector_.addDescriptor(kf_id, cloud);
    auto kf = keyframe_manager_.getKeyframe(kf_id);
    if (kf) {
        const int submap_radius = std::max(0, config_.rhpd_submap_kf_radius);
        PointCloud::Ptr rhpd_cloud = cloud;
        if (submap_radius > 0) {
            rhpd_cloud = keyframe_manager_.buildCausalSubmapInRootFrame(kf_id, submap_radius, kf_id);
        }
        if (rhpd_cloud && !rhpd_cloud->empty() && config_.rhpd_submap_voxel_size > 1e-4) {
            pcl::VoxelGrid<pcl::PointXYZI> voxel;
            voxel.setLeafSize(config_.rhpd_submap_voxel_size,
                              config_.rhpd_submap_voxel_size,
                              config_.rhpd_submap_voxel_size);
            voxel.setInputCloud(rhpd_cloud);
            auto filtered = pcl::make_shared<PointCloud>();
            voxel.filter(*filtered);
            if (!filtered->empty()) rhpd_cloud = filtered;
        }
        kf->rhpd_descriptor = loop_detector_.addRHPD(kf_id, rhpd_cloud);
    }

    if (kf_id == 0) {
        graph_optimizer_.addPriorFactor(kf_id, pose_odom);
    } else {
        auto prev_kf = keyframe_manager_.getKeyframe(kf_id - 1);
        if (prev_kf) {
            EdgeInfo edge;
            edge.from_id = kf_id - 1;
            edge.to_id = kf_id;
            edge.measurement = prev_kf->pose_odom.inverse() * pose_odom;
            edge.information = Eigen::Matrix<double, 6, 6>::Identity();
            edge.information.block<3, 3>(0, 0) *=
                1.0 / (config_.odom_noise_position * config_.odom_noise_position);
            edge.information.block<3, 3>(3, 3) *=
                1.0 / (config_.odom_noise_rotation * config_.odom_noise_rotation);
            edge.type = EdgeType::ODOMETRY;
            graph_optimizer_.addOdometryEdge(edge);
        }
    }

    graph_optimizer_.incrementalOptimize();

    auto optimized_poses = graph_optimizer_.getOptimizedPoses();
    keyframe_manager_.updateOptimizedPoses(optimized_poses);

    if (graph_optimizer_.hasNode(kf_id)) {
        try {
            result.publish_pose = graph_optimizer_.getOptimizedPose(kf_id);
            result.optimized_pose_available = true;
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to get optimized pose: " << e.what();
        }
    }
    return result;
}

}  // namespace core
}  // namespace n3mapping
