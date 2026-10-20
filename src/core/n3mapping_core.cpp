#include "n3mapping/core/n3mapping_core.h"
#include "loop_closure_async.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/floor_attitude.h"
#include "n3mapping/submap_graph_projection.h"
#include "n3mapping/submap_graph_factor.h"
#include "n3mapping/submap_graph_trial_runtime.h"
#include "n3mapping/static_start_guard.h"
#include "n3mapping/pcl_compat.h"
#include <pcl/common/transforms.h>

namespace n3mapping {

namespace {

std::vector<core::AnchoredDenseTrajectorySample> anchorDenseTrajectory(
    const std::vector<core::DenseTrajectoryPose> &dense_optimized,
    const std::vector<Keyframe::Ptr> &keyframes) {
  std::vector<core::AnchoredDenseTrajectorySample> samples;
  samples.reserve(dense_optimized.size());

  for (const auto &dense_pose : dense_optimized) {
    Keyframe::Ptr anchor;
    double best_time = -std::numeric_limits<double>::infinity();
    for (const auto &kf : keyframes) {
      if (kf && kf->timestamp <= dense_pose.timestamp &&
          kf->timestamp >= best_time) {
        anchor = kf;
        best_time = kf->timestamp;
      }
    }
    if (!anchor && !keyframes.empty()) {
      anchor = keyframes.front();
    }

    core::AnchoredDenseTrajectorySample sample;
    sample.seq = dense_pose.seq;
    sample.timestamp = dense_pose.timestamp;
    sample.pose_world_lidar_raw = dense_pose.pose_world_lidar;
    if (anchor) {
      sample.anchor_keyframe_id = anchor->id;
      sample.anchor_pose_world_lidar_raw = anchor->pose_optimized;
      sample.has_anchor = true;
      sample.use_bracketing_correction = false;
    }
    samples.push_back(std::move(sample));
  }
  return samples;
}

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

core::BackendOutput
N3MappingCore::processMappingFrame(const core::LioFrame &frame) {
  if (!frame.pose_valid || !frame.undistorted_cloud ||
      frame.undistorted_cloud->empty()) {
    auto output = makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    output.mapping_block_reason = core::MappingBlockReason::InvalidInput;
    return output;
  }

  const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
  // Once the front end has run away nothing downstream can use what it sends,
  // and every frame that keeps being added replaces a usable partial map with
  // an unusable whole one. Stop taking them.
  if (config_.odom_sanity_enable) {
    if (!odometry_sanity_configured_) {
      OdometrySanityLimits limits;
      limits.max_speed_mps = config_.odom_sanity_max_speed_mps;
      limits.max_angular_rate_dps = config_.odom_sanity_max_angular_rate_dps;
      limits.max_consecutive_violations = config_.odom_sanity_max_consecutive;
      odometry_sanity_ = OdometrySanity(limits);
      odometry_sanity_configured_ = true;
    }
    if (odometry_sanity_.check(timestamp, frame.T_world_lidar).diverged) {
      auto output = makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
      output.mapping_block_reason = core::MappingBlockReason::OdometryDiverged;
      return output;
    }
  }

  // Do not seed the graph with unsettled odometry. A stable scan/pose window
  // also permits mapping when the robot remains stationary.
  if (config_.mapping_static_start_guard_enable) {
    if (!static_start_guard_configured_) {
      static_start_guard_ =
          StaticStartGuard(staticStartGuardOptionsFromConfig(config_));
      static_start_guard_configured_ = true;
    }
    if (!static_start_guard_.update(timestamp, *frame.undistorted_cloud,
                                   frame.T_world_lidar)) {
      auto output = makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
      output.mapping_block_reason = core::MappingBlockReason::StaticStartGuard;
      return output;
    }
  }

  auto &keyframes = session_->keyframeManager();
  if (!keyframes.shouldAddKeyframe(frame.T_world_lidar)) {
    Eigen::Isometry3d pose_map = frame.T_world_lidar;
    if (const auto latest = keyframes.getLatestKeyframe()) {
      // Non-keyframes use the latest optimized map<-odom anchor, without
      // writing that correction back into the raw odometry.
      pose_map = latest->pose_optimized * latest->pose_odom.inverse() *
                 frame.T_world_lidar;
    }
    if (!external_dense_trajectory_recording_enabled_) {
      appendDenseTrajectorySampleWithLatestAnchor(timestamp,
                                                  frame.T_world_lidar);
    }
    return makeOutput(true, pose_map, frame.undistorted_cloud);
  }

  const int64_t keyframe_id = keyframes.addKeyframe(
      timestamp, frame.T_world_lidar, frame.undistorted_cloud);
  session_->loopDetector().addDescriptor(keyframe_id, frame.undistorted_cloud);
  addRhpdDescriptorForKeyframe(keyframe_id, frame.undistorted_cloud);

  if (keyframe_id == 0) {
    session_->graphOptimizer().addPriorFactor(keyframe_id, frame.T_world_lidar);
  } else {
    addOdometryConstraint(keyframe_id, frame.T_world_lidar);
  }

  // The only absolute attitude the graph ever sees. Measured from this scan's
  // own returns, so it is independent of the pose it constrains.
  if (config_.floor_attitude_enable) {
    FloorNormalOptions floor_options;
    floor_options.max_radius_m = config_.floor_attitude_max_radius_m;
    floor_options.min_points = config_.floor_attitude_min_points;
    const auto floor = estimateFloorNormal(*frame.undistorted_cloud,
                                           frame.T_world_lidar, floor_options);
    if (floor.valid) {
      session_->graphOptimizer().addFloorAttitudeFactor(keyframe_id,
                                                        floor.normal_body);
      ++floor_attitude_accepted_;
    } else {
      ++floor_attitude_rejected_;
    }
  }

  session_->graphOptimizer().incrementalOptimize();
  refreshOptimizedPoses();

  if (session_->submapBuilder().enabled()) {
    const auto committed_keyframe = keyframes.getKeyframe(keyframe_id);
    if (!session_->submapBuilder().appendKeyframe(committed_keyframe)) {
      std::cerr << "Shadow submap append failed for committed keyframe id="
                << keyframe_id << '\n';
    } else {
      // graph_update necessarily runs before the just-committed keyframe can
      // join a shadow submap. Refresh once more so topology diagnostics see
      // complete ownership rather than remaining one keyframe behind.
      refreshSubmapPoses("submap_append");
    }
  }

  Eigen::Isometry3d optimized_pose = frame.T_world_lidar;
  if (session_->graphOptimizer().hasNode(keyframe_id)) {
    try {
      optimized_pose = session_->graphOptimizer().getOptimizedPose(keyframe_id);
    } catch (const std::exception &) {
      optimized_pose = frame.T_world_lidar;
    }
  }

  auto output = makeOutput(true, optimized_pose, frame.undistorted_cloud);
  output.accepted_keyframe = true;
  output.keyframe_id = keyframe_id;
  if (!external_dense_trajectory_recording_enabled_) {
    if (auto kf = keyframes.getKeyframe(keyframe_id)) {
      appendDenseTrajectorySample(timestamp, frame.T_world_lidar, keyframe_id,
                                  kf->pose_odom);
    }
  }

  {
    std::lock_guard<std::mutex> lock(loop_queue_mutex_);
    loop_detection_queue_.push_back(keyframe_id);
  }
  return output;
}

core::BackendOutput
N3MappingCore::processMapExtensionFrame(const core::LioFrame &frame) {
  TrackingPerformance tracking_performance;
  using PerformanceClock = std::chrono::steady_clock;
  core::BackendPerformanceTiming performance;
  performance.enabled = config_.reloc_debug_enable;
  const auto timingStarted = [&]() {
    return performance.enabled ? PerformanceClock::now()
                               : PerformanceClock::time_point{};
  };
  const auto elapsedMilliseconds = [&](PerformanceClock::time_point started) {
    return std::chrono::duration<double, std::milli>(PerformanceClock::now() -
                                                     started)
        .count();
  };
  const auto make_output = [&](bool success, const Eigen::Isometry3d &pose,
                               const PointCloud::Ptr &cloud) {
    auto output = makeOutput(success, pose, cloud);
    output.tracking_performance = tracking_performance;
    if (performance.enabled) {
      output.performance = performance;
    }
    return output;
  };

  if (!frame.pose_valid || !frame.undistorted_cloud ||
      frame.undistorted_cloud->empty() || !map_loaded_) {
    return make_output(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  auto &resuming = session_->mappingResuming();
  auto &localizer = session_->worldLocalizing();
  const auto state = resuming.getState();

  if (state == MappingResumingState::MAP_LOADED) {
    const auto relocalization_started = timingStarted();
    RelocalizationPerformance relocalization_performance;
    const bool locked = resuming.performInitialRelocalization(
        frame.undistorted_cloud, frame.T_world_lidar,
        frame.source_frame_id, &relocalization_performance);
    if (performance.enabled) {
      performance.initial_relocalization_ms =
          elapsedMilliseconds(relocalization_started);
    }
    auto output = make_output(
        locked, localizer.getMapToOdomTransform() * frame.T_world_lidar,
        frame.undistorted_cloud);
    output.relocalization_locked = locked;
    output.relocalization_performance = relocalization_performance;
    output.relocalization_state =
        locked ? RelocalizationState::FULL_6DOF_LOCKED
               : RelocalizationState::SEARCHING;
    output.pose_source =
        locked ? PoseSource::GEOMETRICALLY_CORRECTED : PoseSource::NONE;
    output.relocalization_decision =
        locked ? "accepted" : "initial_relocalization_rejected";
    if (locked) {
      const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
      const int64_t matched_id = localizer.getLastMatchedKeyframeId();
      output.relocalization_seed_keyframe_id = matched_id;
      output.relocalization_support_keyframe_id = matched_id;
      output.matched_keyframe_id = matched_id;
      auto matched_kf = session_->keyframeManager().getKeyframe(matched_id);
      if (!external_dense_trajectory_recording_enabled_ && matched_kf) {
        appendDenseTrajectorySample(timestamp, output.T_world_lidar, matched_id,
                                    matched_kf->pose_optimized, false);
      }
    }
    return output;
  }

  if (state != MappingResumingState::RELOCALIZED &&
      state != MappingResumingState::EXTENDING) {
    return make_output(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  // While old-map geometry is visible, extension must track that immutable
  // geometry continuously. A constant relocalization transform only carries
  // the session LIO's time-varying drift into the resumed map. This strict
  // path never falls back to a new global search or to self-created frames.
  // Loss suppresses output and graph writes, but does not disable subsequent
  // loaded-map registration when usable geometry returns.
  performance.tracking_attempted = true;
  const auto tracking_started = timingStarted();
  const RelocResult tracking = localizer.trackLoadedMap(
      frame.undistorted_cloud, frame.T_world_lidar);
  tracking_performance = tracking.tracking_performance;
  if (performance.enabled) {
    performance.loaded_map_tracking_ms =
        elapsedMilliseconds(tracking_started);
  }
  if (!tracking.success ||
      tracking.state != RelocalizationState::FULL_6DOF_LOCKED ||
      tracking.pose_source != PoseSource::GEOMETRICALLY_CORRECTED ||
      tracking.matched_keyframe_id < 0) {
    auto output = make_output(
        false, tracking.pose_in_map, frame.undistorted_cloud);
    output.relocalization_state = tracking.state;
    output.pose_source = tracking.pose_source;
    output.relocalization_decision = tracking.decision;
    output.relocalization_seed_keyframe_id = tracking.seed_keyframe_id;
    output.relocalization_support_keyframe_id =
        tracking.support_keyframe_id;
    output.matched_keyframe_id = tracking.matched_keyframe_id;
    return output;
  }

  Eigen::Isometry3d pose_map = tracking.pose_in_map;
  const auto keyframe_gate_started = timingStarted();
  const bool should_add_keyframe =
      resuming.shouldAddKeyframe(frame.T_world_lidar);
  if (performance.enabled) {
    performance.keyframe_gate_ms =
        elapsedMilliseconds(keyframe_gate_started);
  }
  if (!should_add_keyframe) {
    if (!external_dense_trajectory_recording_enabled_) {
      const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
      auto matched_kf = session_->keyframeManager().getKeyframe(
          tracking.matched_keyframe_id);
      if (matched_kf && matched_kf->is_from_loaded_map) {
        appendDenseTrajectorySample(timestamp, pose_map, matched_kf->id,
                                    matched_kf->pose_optimized, false);
      }
    }
    auto output = make_output(true, pose_map, frame.undistorted_cloud);
    output.relocalization_state = tracking.state;
    output.pose_source = tracking.pose_source;
    output.relocalization_decision = tracking.decision;
    output.relocalization_seed_keyframe_id = tracking.seed_keyframe_id;
    output.relocalization_support_keyframe_id =
        tracking.support_keyframe_id;
    output.matched_keyframe_id = tracking.matched_keyframe_id;
    return output;
  }

  const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
  MappingResumingTiming resuming_timing;
  const int64_t keyframe_id = resuming.processNewKeyframe(
      timestamp, frame.T_world_lidar, frame.undistorted_cloud,
      tracking.matched_keyframe_id, tracking.pose_in_map,
      performance.enabled ? &resuming_timing : nullptr);
  if (performance.enabled) {
    performance.keyframe_commit_ms = resuming_timing.total_ms;
    performance.graph_update_ms = resuming_timing.graph_update_ms;
    performance.descriptor_update_ms = resuming_timing.descriptor_update_ms;
  }
  if (keyframe_id >= 0) {
    // The strict tracker already supplied a loaded-map constraint for this
    // keyframe (the first one is represented by SESSION_ANCHOR). Running the
    // global descriptor pipeline again would be redundant and can only add a
    // less local alias.
    const auto refresh_started = timingStarted();
    refreshOptimizedPoses();
    if (performance.enabled) {
      performance.post_commit_refresh_ms =
          elapsedMilliseconds(refresh_started);
    }
    if (session_->graphOptimizer().hasNode(keyframe_id)) {
      try {
        pose_map = session_->graphOptimizer().getOptimizedPose(keyframe_id);
      } catch (const std::exception &) {
        pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
      }
    }
  }

  auto output = make_output(keyframe_id >= 0, pose_map,
                            frame.undistorted_cloud);
  output.accepted_keyframe = keyframe_id >= 0;
  output.keyframe_id = keyframe_id;
  output.relocalization_state = tracking.state;
  output.pose_source = tracking.pose_source;
  output.relocalization_decision = tracking.decision;
  output.relocalization_seed_keyframe_id = tracking.seed_keyframe_id;
  output.relocalization_support_keyframe_id = tracking.support_keyframe_id;
  output.matched_keyframe_id = tracking.matched_keyframe_id;
  if (!external_dense_trajectory_recording_enabled_ && keyframe_id >= 0) {
    if (auto kf = session_->keyframeManager().getKeyframe(keyframe_id)) {
      appendDenseTrajectorySample(timestamp, kf->pose_odom, keyframe_id,
                                  kf->pose_odom, false);
    }
  }
  return output;
}

bool N3MappingCore::loadMap(const std::string &map_path) {
  try {
    auto candidate = std::make_unique<core::N3MappingSession>(config_);
    std::vector<core::DenseTrajectoryPose> loaded_dense_optimized;
    core::DenseTrajectoryMetadata loaded_dense_metadata;
    if (!candidate->mapSerializer().loadMap(
            map_path, candidate->keyframeManager(), candidate->loopDetector(),
            candidate->graphOptimizer(), &loaded_dense_optimized,
            &loaded_dense_metadata, &candidate->submapBuilder())) {
      return false;
    }
    if (!candidate->mappingResuming().initializeFromLoadedMap()) {
      return false;
    }

    std::string atlas_error;
    if (!candidate->worldLocalizing().loadLocalizationAtlas(map_path,
                                                            &atlas_error)) {
      std::cerr << "Failed to load localization atlas: " << atlas_error << '\n';
      return false;
    }

    auto loaded_dense_samples = anchorDenseTrajectory(
        loaded_dense_optimized,
        candidate->keyframeManager().getAllKeyframes());

    // Every fallible map-specific step completed against the candidate. The
    // session owns all cross-referenced components, so one pointer swap is the
    // transaction commit and cannot leave mixed revisions behind.
    session_.swap(candidate);
    dense_trajectory_samples_.swap(loaded_dense_samples);
    dense_trajectory_metadata_ = std::move(loaded_dense_metadata);
    {
      std::lock_guard<std::mutex> lock(loop_queue_mutex_);
      loop_detection_queue_.clear();
    }
    last_loop_check_id_ = -1000;
    loop_count_ = 0;
    floor_attitude_accepted_ = 0;
    floor_attitude_rejected_ = 0;
    localization_source_frame_id_.clear();
    map_loaded_ = true;
    return true;
  } catch (const std::exception &error) {
    std::cerr << "Failed to prepare map transaction: " << error.what() << '\n';
    return false;
  }
}

bool N3MappingCore::saveMap(const std::string &map_path) {
  if (async_loops_) processPendingLoopClosures();
  if (!refreshSubmapPoses("save_map")) {
    return false;
  }
  const auto dense_optimized_trajectory = buildDenseOptimizedTrajectory();
  core::DenseTrajectoryMetadata metadata = dense_trajectory_metadata_;
  if (!dense_optimized_trajectory.empty() &&
      (metadata.source.empty() || metadata.source == "none")) {
    metadata.source = "native";
    metadata.degraded = false;
  }
  return session_->mapSerializer().saveMap(
      map_path, session_->keyframeManager(), session_->loopDetector(),
      session_->graphOptimizer(), dense_optimized_trajectory, metadata,
      &session_->submapBuilder());
}

bool N3MappingCore::saveGlobalMap(const std::string &pcd_path) {
  return session_->mapSerializer().saveGlobalMap(
      pcd_path, session_->keyframeManager(),
      config_.save_global_map_voxel_size);
}

bool N3MappingCore::saveMapSnapshot(std::string *error) {
  if (session_->keyframeManager().size() < 1) {
    if (error)
      *error = "no_keyframes";
    return false;
  }

  const std::string map_file = config_.map_save_path + "/n3map.pbstream";
  if (!saveMap(map_file)) {
    if (error)
      *error = "save_pbstream_failed";
    return false;
  }

  if (config_.save_global_map_on_shutdown) {
    const std::string global_map_file =
        config_.map_save_path + "/global_map.pcd";
    if (!saveGlobalMap(global_map_file)) {
      if (error)
        *error = "save_global_map_failed";
      return false;
    }
  }

  return true;
}

core::LioFrame::PointCloud::Ptr N3MappingCore::buildGlobalMap() const {
  return session_->mapSerializer().buildGlobalMap(
      session_->keyframeManager(), config_.global_map_voxel_size);
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

void N3MappingCore::appendDenseTrajectorySample(
    double timestamp, const Eigen::Isometry3d &raw_pose,
    int64_t anchor_keyframe_id, const Eigen::Isometry3d &anchor_raw_pose,
    bool use_bracketing_correction) {
  core::AnchoredDenseTrajectorySample sample;
  sample.seq = static_cast<uint64_t>(dense_trajectory_samples_.size());
  sample.timestamp = timestamp;
  sample.pose_world_lidar_raw = raw_pose;
  sample.anchor_keyframe_id = anchor_keyframe_id;
  sample.anchor_pose_world_lidar_raw = anchor_raw_pose;
  sample.has_anchor = anchor_keyframe_id >= 0;
  sample.use_bracketing_correction = use_bracketing_correction;
  if (dense_trajectory_metadata_.source == "keyframe_fallback") {
    dense_trajectory_metadata_.source = "mixed_keyframe_fallback_and_high_rate";
    dense_trajectory_metadata_.degraded = true;
  } else if (dense_trajectory_metadata_.source.empty() ||
             dense_trajectory_metadata_.source == "none") {
    dense_trajectory_metadata_.source = "native";
    dense_trajectory_metadata_.degraded = false;
  }
  dense_trajectory_samples_.push_back(sample);
}

void N3MappingCore::appendDenseTrajectorySampleWithLatestAnchor(
    double timestamp, const Eigen::Isometry3d &raw_pose,
    bool use_bracketing_correction) {
  auto latest = session_->keyframeManager().getLatestKeyframe();
  if (!latest) {
    appendDenseTrajectorySample(timestamp, raw_pose, -1,
                                Eigen::Isometry3d::Identity(),
                                use_bracketing_correction);
    return;
  }

  const Eigen::Isometry3d anchor_raw_pose =
      latest->is_from_loaded_map ? latest->pose_optimized : latest->pose_odom;
  appendDenseTrajectorySample(timestamp, raw_pose, latest->id, anchor_raw_pose,
                              use_bracketing_correction);
}

std::vector<core::DenseTrajectoryPose>
N3MappingCore::buildDenseOptimizedTrajectory() const {
  std::vector<core::DenseTrajectoryPose> dense_optimized;
  dense_optimized.reserve(dense_trajectory_samples_.size());
  std::vector<Keyframe::Ptr> time_index;
  if (std::any_of(dense_trajectory_samples_.begin(), dense_trajectory_samples_.end(),
                  [](const auto& sample) { return sample.use_bracketing_correction; })) {
    time_index = session_->keyframeManager().getAllKeyframes();
    time_index.erase(std::remove_if(time_index.begin(), time_index.end(),
        [](const auto& kf) { return !kf || std::isnan(kf->timestamp); }), time_index.end());
    std::stable_sort(time_index.begin(), time_index.end(),
        [](const auto& lhs, const auto& rhs) { return lhs->timestamp < rhs->timestamp; });
    // The scan-based implementation retained the first keyframe at each time.
    // Preserve that rule, including legacy overlapping session timestamps.
    time_index.erase(std::unique(time_index.begin(), time_index.end(),
        [](const auto& lhs, const auto& rhs) { return lhs->timestamp == rhs->timestamp; }),
        time_index.end());
  }
  for (const auto &sample : dense_trajectory_samples_) {
    core::DenseTrajectoryPose pose;
    pose.seq = sample.seq;
    pose.timestamp = sample.timestamp;
    pose.pose_world_lidar = sample.pose_world_lidar_raw;
    if (sample.use_bracketing_correction) {
      pose.pose_world_lidar = interpolateDenseCorrection(sample.timestamp, time_index) *
                              sample.pose_world_lidar_raw;
    } else if (sample.has_anchor) {
      auto anchor =
          session_->keyframeManager().getKeyframe(sample.anchor_keyframe_id);
      if (anchor) {
        pose.pose_world_lidar = anchor->pose_optimized *
                                sample.anchor_pose_world_lidar_raw.inverse() *
                                sample.pose_world_lidar_raw;
      }
    }
    dense_optimized.push_back(pose);
  }
  return dense_optimized;
}

Eigen::Isometry3d
N3MappingCore::interpolateDenseCorrection(
    double timestamp, const std::vector<Keyframe::Ptr>& time_index) const {
  if (time_index.empty() || std::isnan(timestamp)) {
    return Eigen::Isometry3d::Identity();
  }
  const auto next = std::lower_bound(time_index.begin(), time_index.end(), timestamp,
      [](const auto& kf, double time) { return kf->timestamp < time; });
  const auto after = next == time_index.end() ? time_index.back() : *next;
  const auto before = next == time_index.begin() || after->timestamp == timestamp
                          ? after : *std::prev(next);

  const Eigen::Isometry3d correction_before =
      before->pose_optimized * before->pose_odom.inverse();
  const Eigen::Isometry3d correction_after =
      after->pose_optimized * after->pose_odom.inverse();
  double alpha = 0.0;
  const double dt = after->timestamp - before->timestamp;
  if (std::isfinite(dt) && dt > 1e-9) {
    alpha = std::clamp((timestamp - before->timestamp) / dt, 0.0, 1.0);
  }

  Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
  correction.translation() = (1.0 - alpha) * correction_before.translation() +
                             alpha * correction_after.translation();
  Eigen::Quaterniond qb(correction_before.rotation());
  Eigen::Quaterniond qa(correction_after.rotation());
  qb.normalize();
  qa.normalize();
  correction.linear() = qb.slerp(alpha, qa).normalized().toRotationMatrix();
  return correction;
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
        keyframe_id, submap_radius, keyframe_id);
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
    if (!trial.persisted) {
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
              << " persisted=" << trial.persisted
              << " failure_reason="
              << (trial.diagnostics.failure_reason.empty()
                      ? "none"
                      : trial.diagnostics.failure_reason);
  }
  return true;
}

} // namespace n3mapping
