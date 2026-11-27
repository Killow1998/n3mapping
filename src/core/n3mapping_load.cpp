#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping_session.h"
#include "n3mapping/async_log.h"
#include "n3mapping/n3map_proto_utils.h"
#include "n3map.pb.h"
#include <limits>
#include <utility>
#include <glog/logging.h>

namespace n3mapping {
namespace {

std::vector<core::AnchoredDenseTrajectorySample> anchorDenseTrajectory(
    const std::vector<core::DenseTrajectoryPose> &dense_optimized,
    const std::vector<Keyframe::Ptr> &keyframes) {
  std::vector<core::AnchoredDenseTrajectorySample> samples;
  samples.reserve(dense_optimized.size());

  for (const auto &dense_pose : dense_optimized) {
    if (dense_pose.correction_source) {
      samples.push_back(*dense_pose.correction_source);
      continue;
    }
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

}

bool N3MappingCore::loadMap(const std::string &map_path) {
  LogOperation operation("map_load", map_path);
  try {
    auto candidate = std::make_unique<core::N3MappingSession>(config_);
    std::vector<core::DenseTrajectoryPose> loaded_dense_optimized;
    core::DenseTrajectoryMetadata loaded_dense_metadata;
    PbstreamLoadOptions options;
    options.allow_keyframe_fallback_dense = true;
    options.rebuild_descriptors = false;
    if (!candidate->serializer.loadMap(
            map_path, candidate->keyframes, candidate->detector,
            candidate->optimizer, &loaded_dense_optimized,
            &loaded_dense_metadata, options, &candidate->submaps)) {
      return false;
    }
    if (!candidate->resuming.initializeFromLoadedMap()) {
      return false;
    }

    std::string atlas_error;
    if (!candidate->localizer.loadLocalizationAtlas(map_path,
                                                            &atlas_error, &candidate->detector)) {
      LOG(ERROR) << "Failed to load localization atlas: " << atlas_error << '\n';
      return false;
    }

    auto loaded_dense_samples = anchorDenseTrajectory(
        loaded_dense_optimized,
        candidate->keyframes.getAllKeyframes());

    std::string compact_map_path;
    if (parseCoreRunMode(config_.mode) == CoreRunMode::LOCALIZATION &&
        candidate->localizer.compactLoadedKeyframeClouds()) {
      compact_map_path = map_path;
    }
    session_.swap(candidate);
    compact_localization_map_path_.swap(compact_map_path);
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
    extension_anchor_frame_.reset();
    odometry_sanity_.reset();
    map_loaded_ = true;
    operation.succeed();
    return true;
  } catch (const std::exception &error) {
    LOG(ERROR) << "Failed to prepare map transaction: " << error.what() << '\n';
    return false;
  }
}

bool N3MappingCore::restoreLocalizationKeyframeClouds() {
  if (compact_localization_map_path_.empty()) return true;
  std::string error;
  const auto hash = LocalizationAtlas::fileSha256(compact_localization_map_path_, &error);
  if (hash.empty() || hash != session_->localizer.atlasSourceMapSha256()) {
    LOG(ERROR) << "The loaded localization map changed or became unreadable: " << error;
    return false;
  }
  N3Map map;
  if (!readN3MapProtoFromFile(compact_localization_map_path_, &map, &error)) {
    LOG(ERROR) << "Cannot restore the localization map keyframes: " << error;
    return false;
  }
  std::map<int64_t, Keyframe::PointCloudT::Ptr> clouds;
  for (const auto& frame : map.keyframes()) {
    Keyframe::PointCloudT::Ptr cloud;
    if (!pointCloudFromProto(frame.cloud(), &cloud, &error)) {
      LOG(ERROR) << "Cannot restore localization map keyframe " << frame.id() << ": " << error;
      return false;
    }
    clouds.emplace(frame.id(), std::move(cloud));
  }
  session_->keyframes.replaceKeyframeClouds(clouds);
  compact_localization_map_path_.clear();
  return true;
}

core::LioFrame::PointCloud::ConstPtr N3MappingCore::compactLocalizationCloud() const {
  return compact_localization_map_path_.empty() ? nullptr : session_->localizer.atlasGlobalMap();
}

}
