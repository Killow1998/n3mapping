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

}

N3MappingCore::N3MappingCore(const Config &config,
                             WorldCloudOutput world_cloud_output)
    : config_(validateOrThrow(config)),
      world_cloud_output_(world_cloud_output),
      session_(std::make_unique<core::N3MappingSession>(config_)),
      odometry_sanity_({config_.odom_sanity_max_speed_mps,
                        config_.odom_sanity_max_angular_rate_dps,
                        config_.odom_sanity_max_consecutive}),
      static_start_guard_(staticStartGuardOptionsFromConfig(config_)) {}

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
  if (world_cloud_output_ == WorldCloudOutput::Enabled) {
    output.cloud_world = pcl::make_shared<PointCloud>();
    if (cloud && !cloud->empty())
      pcl::transformPointCloud(*cloud, *output.cloud_world,
                               pose.matrix().cast<float>());
  }
  return output;
}

}
