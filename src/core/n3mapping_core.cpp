#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/n3mapping_session.h"
#include "n3mapping/n3map_proto_utils.h"
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
#include "n3mapping/submap_graph_projection.h"
#include "n3mapping/submap_graph_factor.h"
#include "n3mapping/submap_graph_trial_runtime.h"
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
  return session_->keyframeManager().getKeyframe(id);
}

std::vector<Keyframe::Ptr> N3MappingCore::getAllKeyframes() const {
  return session_->keyframeManager().getAllKeyframes();
}

KeyframeMapRevision N3MappingCore::mapRevision() const {
  return session_->keyframeManager().revision();
}

std::map<int64_t, Eigen::Isometry3d> N3MappingCore::getOptimizedPoses() const {
  return session_->graphOptimizer().getOptimizedPoses();
}

std::vector<core::DenseTrajectoryPose>
N3MappingCore::getDenseOptimizedTrajectory() const {
  return buildDenseOptimizedTrajectory();
}

void N3MappingCore::setExternalDenseTrajectoryRecordingEnabled(bool enabled) {
  external_dense_trajectory_recording_enabled_ = enabled;
}

void N3MappingCore::recordDenseTrajectoryPose(
    CoreRunMode mode, double timestamp,
    const Eigen::Isometry3d &pose_world_lidar) {
  if (!external_dense_trajectory_recording_enabled_ ||
      mode != CoreRunMode::MAPPING || !std::isfinite(timestamp) ||
      !isFinitePose(pose_world_lidar)) {
    return;
  }

  appendDenseTrajectorySampleWithLatestAnchor(timestamp, pose_world_lidar);
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
  auto kf = session_->keyframeManager().getKeyframe(keyframe_id);
  if (!kf) {
    return;
  }

  const int submap_radius = std::max(0, config_.rhpd_submap_kf_radius);
  PointCloud::Ptr rhpd_cloud = fallback_cloud;
  if (submap_radius > 0) {
    rhpd_cloud = session_->keyframeManager().buildCausalSubmapInRootFrame(
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

  session_->loopDetector().addRHPD(keyframe_id, rhpd_cloud);
}

void N3MappingCore::refreshOptimizedPoses(const char* context) {
  session_->keyframeManager().updateOptimizedPoses(
      session_->graphOptimizer().getOptimizedPoses());
  refreshSubmapPoses(context);
}

bool N3MappingCore::refreshSubmapPoses(const char* context) {
  auto& submaps = session_->submapBuilder();
  if (!submaps.enabled()) {
    return true;
  }
  const auto keyframes = session_->keyframeManager().getAllKeyframes();
  const auto diagnostics = submaps.refreshMapPoses(keyframes);
  if (!diagnostics.valid) {
    LOG(WARNING) << "[SubmapShadow] pose projection refresh failed context="
                 << (context ? context : "unknown")
                 << " reason=" << diagnostics.failure_reason;
    return false;
  }
  VLOG(1) << "[SubmapShadow] pose projection context="
          << (context ? context : "unknown")
          << " submaps=" << diagnostics.submap_count
          << " projected_keyframes="
          << diagnostics.projected_keyframe_count
          << " unassigned_keyframes="
          << diagnostics.unassigned_keyframe_count
          << " refreshed_submaps="
          << diagnostics.refreshed_submap_count
          << " mean_translation_residual_m="
          << diagnostics.mean_translation_residual_m
          << " max_translation_residual_m="
          << diagnostics.max_translation_residual_m
          << " mean_rotation_residual_rad="
          << diagnostics.mean_rotation_residual_rad
          << " max_rotation_residual_rad="
          << diagnostics.max_rotation_residual_rad;

  const auto graph_snapshot = buildSubmapGraphSnapshot(
      submaps.getSubmaps(), keyframes, session_->graphOptimizer().getEdges(),
      session_->graphOptimizer().floorAttitudeConstraints());
  if (!graph_snapshot.valid) {
    // SG-02 remains observational: an invalid shadow snapshot is visible but
    // cannot change the reference keyframe-graph lifecycle.
    LOG(WARNING) << "[SubmapGraphShadow] snapshot failed context="
                 << (context ? context : "unknown")
                 << " reason=" << graph_snapshot.failure_reason;
  } else {
    VLOG(1) << "[SubmapGraphShadow] snapshot context="
            << (context ? context : "unknown")
            << " nodes=" << graph_snapshot.nodes.size()
            << " source_edges=" << graph_snapshot.source_edge_count
            << " intra_edges=" << graph_snapshot.intra_submap_edge_count
            << " cross_edges=" << graph_snapshot.cross_submap_edge_count
            << " unassigned_edges="
            << graph_snapshot.unassigned_endpoint_edge_count
            << " floor_assigned="
            << graph_snapshot.assigned_floor_constraint_count
            << " floor_unassigned="
            << graph_snapshot.unassigned_floor_constraint_count
            << " max_cross_translation_residual_m="
            << graph_snapshot.max_cross_edge_translation_residual_m
            << " max_cross_rotation_residual_rad="
            << graph_snapshot.max_cross_edge_rotation_residual_rad;
    const auto topology = evaluateSubmapGraphTopology(graph_snapshot);
    if (!topology.valid) {
      LOG(WARNING) << "[SubmapGraphShadow] topology failed context="
                   << (context ? context : "unknown")
                   << " reason=" << topology.failure_reason;
    } else {
      VLOG(1) << "[SubmapGraphShadow] topology context="
              << (context ? context : "unknown")
              << " nodes=" << topology.node_count
              << " components=" << topology.component_count
              << " isolated_submaps=" << topology.isolated_submap_count
              << " cross_edges=" << topology.cross_edge_count
              << " cross_session_edges="
              << topology.cross_session_edge_count
              << " keyframe_ownership_complete="
              << topology.keyframe_ownership_complete
              << " constraint_coverage_complete="
              << topology.constraint_coverage_complete
              << " connected=" << topology.connected
              << " shadow_graph_ready=" << topology.shadow_graph_ready;
    }
    const auto factors = evaluateSubmapGraphFactors(graph_snapshot);
    if (!factors.valid) {
      LOG(WARNING) << "[SubmapGraphShadow] factor semantics failed context="
                   << (context ? context : "unknown")
                   << " reason=" << factors.failure_reason;
    } else {
      VLOG(1) << "[SubmapGraphShadow] factor semantics context="
              << (context ? context : "unknown")
              << " cross_edges=" << factors.cross_edge_count
              << " full_6d_edges=" << factors.full_6d_edge_count
              << " xy_yaw_lifted_only="
              << factors.xy_yaw_exact_lifted_only_count
              << " direct_between_residual_equivalent="
              << factors.direct_between_residual_equivalent_count
              << " information_transportable="
              << factors.direct_between_information_transportable_count
              << " information_fallback_required="
              << factors.information_fallback_required_count
              << " assigned_floor_factors="
              << factors.assigned_floor_factor_count
              << " max_residual_transport_error="
              << factors.max_residual_transport_error_norm
              << " max_mahalanobis_squared_delta="
              << factors.max_mahalanobis_squared_delta
              << " max_floor_residual_error="
              << factors.max_floor_residual_error_norm;
    }
  }

  const std::string context_name = context ? context : "unknown";
  const auto trial = runSubmapGraphTrialCheckpoint(
      graph_snapshot, config_, "core", context_name);
  if (trial.checkpoint) {
    if (!trial.log_enqueued) {
      LOG(WARNING) << "[SubmapGraphShadow] trial evidence write failed context="
                   << context_name << " path=" << trial.output_path;
    }
    LOG(INFO) << "[SubmapGraphShadow] trial checkpoint context="
              << context_name
              << " valid=" << trial.diagnostics.valid
              << " attempted=" << trial.diagnostics.attempted
              << " solved=" << trial.diagnostics.solved
              << " nodes=" << trial.diagnostics.node_count
              << " active_factors="
              << trial.diagnostics.active_edge_factor_count
              << " initial_error="
              << trial.diagnostics.initial_nonlinear_error
              << " final_error="
              << trial.diagnostics.final_nonlinear_error
              << " max_translation_delta_m="
              << trial.diagnostics.max_translation_delta_m
              << " max_rotation_delta_rad="
              << trial.diagnostics.max_rotation_delta_rad
              << " optimized_keyframe_p95_translation_error_m="
              << trial.diagnostics.optimized_keyframe_comparison
                     .p95_translation_error_m
              << " optimized_keyframe_max_translation_error_m="
              << trial.diagnostics.optimized_keyframe_comparison
                     .max_translation_error_m
              << " optimized_keyframe_p95_rotation_error_rad="
              << trial.diagnostics.optimized_keyframe_comparison
                     .p95_rotation_error_rad
              << " log_enqueued=" << trial.log_enqueued
              << " failure_reason="
              << (trial.diagnostics.failure_reason.empty()
                      ? "none"
                      : trial.diagnostics.failure_reason);
  }
  return true;
}

} // namespace n3mapping
