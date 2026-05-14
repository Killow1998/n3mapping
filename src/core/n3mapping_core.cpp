#include "n3mapping/core/n3mapping_core.h"

#include <algorithm>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/memory.h>

namespace n3mapping {

N3MappingCore::N3MappingCore(const Config& config)
  : config_(config)
  , session_(std::make_unique<core::N3MappingSession>(config_))
{
}

N3MappingCore::~N3MappingCore() = default;

core::BackendOutput N3MappingCore::processMappingFrame(const core::LioFrame& frame)
{
    if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    auto output = makeOutput(true, frame.T_world_lidar, frame.undistorted_cloud);
    auto& keyframes = session_->keyframeManager();
    if (!keyframes.shouldAddKeyframe(frame.T_world_lidar)) {
        return output;
    }

    const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
    const int64_t keyframe_id = keyframes.addKeyframe(timestamp, frame.T_world_lidar, frame.undistorted_cloud);
    session_->loopDetector().addDescriptor(keyframe_id, frame.undistorted_cloud);
    addRhpdDescriptorForKeyframe(keyframe_id, frame.undistorted_cloud);

    if (keyframe_id == 0) {
        session_->graphOptimizer().addPriorFactor(keyframe_id, frame.T_world_lidar);
    } else {
        addOdometryConstraint(keyframe_id, frame.T_world_lidar);
    }

    session_->graphOptimizer().incrementalOptimize();
    refreshOptimizedPoses();

    Eigen::Isometry3d optimized_pose = frame.T_world_lidar;
    if (session_->graphOptimizer().hasNode(keyframe_id)) {
        try {
            optimized_pose = session_->graphOptimizer().getOptimizedPose(keyframe_id);
        } catch (const std::exception&) {
            optimized_pose = frame.T_world_lidar;
        }
    }

    output.accepted_keyframe = true;
    output.keyframe_id = keyframe_id;
    output.T_world_lidar = optimized_pose;
    output.cloud_world = makeWorldCloud(frame.undistorted_cloud, optimized_pose);
    return output;
}

core::BackendOutput N3MappingCore::processLocalizationFrame(const core::LioFrame& frame)
{
    if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    if (!map_loaded_) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    Eigen::Isometry3d pose_map = frame.T_world_lidar;
    bool success = false;
    bool relocalization_locked = false;
    auto& localizer = session_->worldLocalizing();

    if (localizer.isRelocalized()) {
        auto result = localizer.trackLocalization(frame.undistorted_cloud, frame.T_world_lidar);
        if (result.success) {
            pose_map = result.pose_in_map;
            success = true;
        }
    }

    if (!localizer.isRelocalized() || !success) {
        auto result = localizer.relocalize(frame.undistorted_cloud, frame.T_world_lidar);
        if (result.success) {
            pose_map = result.pose_in_map;
            success = true;
            relocalization_locked = true;
        }
    }

    if (!success) {
        pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
    }

    auto output = makeOutput(success, pose_map, frame.undistorted_cloud);
    output.relocalization_locked = relocalization_locked;
    return output;
}

core::BackendOutput N3MappingCore::processMapExtensionFrame(const core::LioFrame& frame)
{
    if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty() || !map_loaded_) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    auto& resuming = session_->mappingResuming();
    auto& localizer = session_->worldLocalizing();
    const auto state = resuming.getState();

    if (state == MappingResumingState::MAP_LOADED) {
        const bool locked = resuming.performInitialRelocalization(frame.undistorted_cloud, frame.T_world_lidar);
        auto output = makeOutput(locked,
                                 localizer.getMapToOdomTransform() * frame.T_world_lidar,
                                 frame.undistorted_cloud);
        output.relocalization_locked = locked;
        return output;
    }

    if (state != MappingResumingState::RELOCALIZED && state != MappingResumingState::EXTENDING) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    Eigen::Isometry3d pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
    if (!session_->keyframeManager().shouldAddKeyframe(pose_map)) {
        return makeOutput(true, pose_map, frame.undistorted_cloud);
    }

    const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
    const int64_t keyframe_id = resuming.processNewKeyframe(timestamp, frame.T_world_lidar, frame.undistorted_cloud);
    if (keyframe_id >= 0) {
        resuming.detectCrossLoops(keyframe_id);
        session_->graphOptimizer().incrementalOptimize();
        refreshOptimizedPoses();
        if (session_->graphOptimizer().hasNode(keyframe_id)) {
            try {
                pose_map = session_->graphOptimizer().getOptimizedPose(keyframe_id);
            } catch (const std::exception&) {
                pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
            }
        }
    }

    auto output = makeOutput(keyframe_id >= 0, pose_map, frame.undistorted_cloud);
    output.accepted_keyframe = keyframe_id >= 0;
    output.keyframe_id = keyframe_id;
    return output;
}

bool N3MappingCore::loadMap(const std::string& map_path)
{
    map_loaded_ = session_->mappingResuming().loadExistingMap(map_path);
    return map_loaded_;
}

bool N3MappingCore::saveMap(const std::string& map_path)
{
    return session_->mapSerializer().saveMap(
        map_path, session_->keyframeManager(), session_->loopDetector(), session_->graphOptimizer());
}

bool N3MappingCore::saveGlobalMap(const std::string& pcd_path)
{
    return session_->mapSerializer().saveGlobalMap(
        pcd_path, session_->keyframeManager(), config_.global_map_voxel_size);
}

bool N3MappingCore::mapLoaded() const
{
    return map_loaded_;
}

std::vector<Keyframe::Ptr> N3MappingCore::getAllKeyframes() const
{
    return session_->keyframeManager().getAllKeyframes();
}

std::map<int64_t, Eigen::Isometry3d> N3MappingCore::getOptimizedPoses() const
{
    return session_->graphOptimizer().getOptimizedPoses();
}

core::BackendOutput N3MappingCore::makeOutput(bool success,
                                              const Eigen::Isometry3d& pose,
                                              const PointCloud::Ptr& cloud) const
{
    core::BackendOutput output;
    output.success = success;
    output.T_world_lidar = pose;
    output.cloud_body = cloud;
    output.cloud_world = makeWorldCloud(cloud, pose);
    return output;
}

N3MappingCore::PointCloud::Ptr N3MappingCore::makeWorldCloud(const PointCloud::Ptr& cloud,
                                                             const Eigen::Isometry3d& pose) const
{
    if (!cloud || cloud->empty()) {
        return pcl::make_shared<PointCloud>();
    }
    auto transformed = pcl::make_shared<PointCloud>();
    pcl::transformPointCloud(*cloud, *transformed, pose.matrix().cast<float>());
    return transformed;
}

void N3MappingCore::addRhpdDescriptorForKeyframe(int64_t keyframe_id, const PointCloud::Ptr& fallback_cloud)
{
    auto kf = session_->keyframeManager().getKeyframe(keyframe_id);
    if (!kf) {
        return;
    }

    const int submap_radius = std::max(0, config_.rhpd_submap_kf_radius);
    PointCloud::Ptr rhpd_cloud = fallback_cloud;
    if (submap_radius > 0) {
        rhpd_cloud = session_->keyframeManager().buildCausalSubmapInRootFrame(keyframe_id, submap_radius, keyframe_id);
    }

    if (rhpd_cloud && !rhpd_cloud->empty() && config_.rhpd_submap_voxel_size > 1e-4) {
        pcl::VoxelGrid<pcl::PointXYZI> voxel;
        voxel.setLeafSize(config_.rhpd_submap_voxel_size,
                          config_.rhpd_submap_voxel_size,
                          config_.rhpd_submap_voxel_size);
        voxel.setInputCloud(rhpd_cloud);
        auto filtered = pcl::make_shared<PointCloud>();
        voxel.filter(*filtered);
        if (!filtered->empty()) {
            rhpd_cloud = filtered;
        }
    }

    kf->rhpd_descriptor = session_->loopDetector().addRHPD(keyframe_id, rhpd_cloud);
}

bool N3MappingCore::addOdometryConstraint(int64_t keyframe_id, const Eigen::Isometry3d& pose)
{
    auto prev_kf = session_->keyframeManager().getKeyframe(keyframe_id - 1);
    if (!prev_kf) {
        return false;
    }

    EdgeInfo edge;
    edge.from_id = keyframe_id - 1;
    edge.to_id = keyframe_id;
    edge.measurement = prev_kf->pose_odom.inverse() * pose;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
    edge.information.block<3, 3>(0, 0) *= 1.0 / (config_.odom_noise_position * config_.odom_noise_position);
    edge.information.block<3, 3>(3, 3) *= 1.0 / (config_.odom_noise_rotation * config_.odom_noise_rotation);
    edge.type = EdgeType::ODOMETRY;
    session_->graphOptimizer().addOdometryEdge(edge);
    return true;
}

void N3MappingCore::refreshOptimizedPoses()
{
    session_->keyframeManager().updateOptimizedPoses(session_->graphOptimizer().getOptimizedPoses());
}

}  // namespace n3mapping
