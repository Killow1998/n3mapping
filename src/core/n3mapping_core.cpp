#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping_session.h"
#include "loop_closure_async.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/pcl_compat.h"
#include <pcl/common/transforms.h>

namespace n3mapping {

namespace {

Config validateOrThrow(const Config &config) {
  std::string error;
  if (!config.validate(&error)) {
    throw std::invalid_argument("Invalid N3MappingCore config: " + error);
  }
  return config;
}

} // namespace

N3MappingCore::N3MappingCore(const Config &config,
                             WorldCloudOutput world_cloud_output)
    : config_(validateOrThrow(config)),
      world_cloud_output_(world_cloud_output),
      session_(std::make_unique<core::N3MappingSession>(config_)) {}

N3MappingCore::~N3MappingCore() = default;

core::BackendOutput N3MappingCore::processFrame(CoreRunMode mode,
                                                const core::LioFrame &frame) {
  switch (mode) {
  case CoreRunMode::LOCALIZATION:
    return processLocalizationFrame(frame);
  case CoreRunMode::MAP_EXTENSION:
    return processMapExtensionFrame(frame);
  case CoreRunMode::MAPPING:
  default:
    return processMappingFrame(frame);
  }
}

bool N3MappingCore::mapLoaded() const { return map_loaded_; }

Keyframe::Ptr N3MappingCore::getKeyframe(int64_t id) const {
  return session_->keyframes.getKeyframe(id);
}

std::vector<Keyframe::Ptr> N3MappingCore::getAllKeyframes() const {
  return session_->keyframes.getAllKeyframes();
}

KeyframeMapRevision N3MappingCore::mapRevision() const {
  return session_->keyframes.revision();
}

std::map<int64_t, Eigen::Isometry3d> N3MappingCore::getOptimizedPoses() const {
  return session_->optimizer.getOptimizedPoses();
}

core::BackendOutput
N3MappingCore::makeOutput(bool success, const Eigen::Isometry3d &pose,
                          const PointCloud::Ptr &cloud) const {
  core::BackendOutput output;
  output.success = success;
  output.T_world_lidar = pose;
  output.cloud_body = cloud;
  output.cloud_world = makeWorldCloud(cloud, pose);
  return output;
}

N3MappingCore::PointCloud::Ptr
N3MappingCore::makeWorldCloud(const PointCloud::Ptr &cloud,
                              const Eigen::Isometry3d &pose) const {
  if (world_cloud_output_ == WorldCloudOutput::Disabled) return nullptr;
  if (!cloud || cloud->empty()) {
    return pcl::make_shared<PointCloud>();
  }
  auto transformed = pcl::make_shared<PointCloud>();
  pcl::transformPointCloud(*cloud, *transformed, pose.matrix().cast<float>());
  return transformed;
}

void N3MappingCore::addRhpdDescriptorForKeyframe(
    int64_t keyframe_id, const PointCloud::Ptr &fallback_cloud) {
  auto kf = session_->keyframes.getKeyframe(keyframe_id);
  if (!kf) {
    return;
  }

  const int submap_radius = std::max(0, config_.rhpd_submap_kf_radius);
  PointCloud::Ptr rhpd_cloud = fallback_cloud;
  if (submap_radius > 0) {
    rhpd_cloud = session_->keyframes.buildCausalSubmapInRootFrame(
        keyframe_id, submap_radius, keyframe_id,
        KeyframeManager::PoseSource::Odometry);
  }

  if (rhpd_cloud && !rhpd_cloud->empty() &&
      config_.rhpd_submap_voxel_size > 1e-4) {
    PointCloud::Ptr filtered;
    if (safeVoxelGridFilter<pcl::PointXYZI>(
            rhpd_cloud, config_.rhpd_submap_voxel_size, &filtered) &&
        filtered && !filtered->empty()) {
      rhpd_cloud = filtered;
    }
  }

  session_->detector.addRHPD(keyframe_id, rhpd_cloud);
}

} // namespace n3mapping
